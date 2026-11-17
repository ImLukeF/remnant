import Foundation

/// Imported research parameters never grant permission to control alarm timing.
public struct SleepModel: Sendable {
    public static let maximumPayloadBytes = 512_000
    fileprivate enum Backing: Sendable {
        case legacy(LinearSleepModel)
        case contextual(ContextualModel)
        case recurrent(RecurrentSleepModel)
    }
    fileprivate let backing: Backing

    public init(data: Data) throws {
        guard data.count <= Self.maximumPayloadBytes else { throw ModelError.invalid }
        struct Header: Decodable { let schemaVersion: Int }
        switch try JSONDecoder().decode(Header.self, from: data).schemaVersion {
        case 1: backing = .legacy(try LinearSleepModel(data: data))
        case 2:
            let model = try JSONDecoder().decode(ContextualModel.self, from: data)
            try model.validate()
            backing = .contextual(model)
        case 3:
            let model = try JSONDecoder().decode(RecurrentSleepModel.self, from: data)
            try model.validate()
            backing = .recurrent(model)
        default: throw ModelError.invalid
        }
    }

    public func makePredictor() -> SleepModelPredictor { SleepModelPredictor(backing: backing) }
    public enum ModelError: Error { case invalid }
}

/// One instance per uninterrupted observation stream. Only current and preceding epochs are retained.
public struct SleepModelPredictor: Sendable {
    private let backing: SleepModel.Backing
    private var frames: [[Double]] = []
    private var previousStart: Date?

    fileprivate init(backing: SleepModel.Backing) { self.backing = backing }

    public mutating func reset() {
        frames.removeAll(keepingCapacity: true)
        previousStart = nil
    }

    public mutating func predict(_ epoch: AudioEpoch) throws -> SleepPrediction {
        guard (epoch.features.count == 7 || epoch.features.count == AudioFeatureExtractor.spectralFeatureNames.count), epoch.features.allSatisfy(\.isFinite),
              epoch.start.timeIntervalSince1970.isFinite, epoch.end.timeIntervalSince1970.isFinite,
              (29...31).contains(epoch.end.timeIntervalSince(epoch.start)) else {
            reset()
            throw SleepModel.ModelError.invalid
        }
        guard epoch.usable else {
            reset()
            return SleepPrediction(start: epoch.start, end: epoch.end, probabilities: [:], usable: false)
        }
        do {
            switch backing {
            case .legacy(let model):
                return try model.predict(AudioEpoch(start: epoch.start, end: epoch.end,
                    features: Array(epoch.features.prefix(7)), usable: epoch.usable))
            case .contextual(let model):
                let dimensions = model.rawFeatureNames.count
                guard epoch.features.count >= dimensions else { throw SleepModel.ModelError.invalid }
                var raw = Array(epoch.features.prefix(dimensions))
                if dimensions > 7 {
                    for index in 7..<dimensions { raw[index] = (raw[index] * 1e6).rounded(.toNearestOrEven) / 1e6 }
                }
                if let previousStart, abs(epoch.start.timeIntervalSince(previousStart) - 30) > 1 { reset() }
                previousStart = epoch.start
                frames.append(raw)
                if frames.count > model.contextEpochs { frames.removeFirst(frames.count - model.contextEpochs) }
                var values = raw
                if model.contextEpochs > 1 {
                    let means = (0..<dimensions).map { k in frames.reduce(0) { $0 + $1[k] } / Double(frames.count) }
                    let deviations = (0..<dimensions).map { k in
                        sqrt(frames.reduce(0) { total, row in
                            let delta = row[k] - means[k]
                            return total + delta * delta
                        } / Double(frames.count))
                    }
                    values += means + deviations + (0..<dimensions).map { raw[$0] - frames[0][$0] }
                }
                let probabilities = try model.probabilities(values)
                return SleepPrediction(start: epoch.start, end: epoch.end,
                    probabilities: Dictionary(uniqueKeysWithValues: zip(model.classes, probabilities)), usable: true)
            case .recurrent(let model):
                guard epoch.features.count == model.rawFeatureNames.count else { throw SleepModel.ModelError.invalid }
                var raw = epoch.features
                for index in 7..<raw.count { raw[index] = (raw[index] * 1e6).rounded(.toNearestOrEven) / 1e6 }
                guard raw.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
                if let previousStart, abs(epoch.start.timeIntervalSince(previousStart) - 30) > 1 { reset() }
                previousStart = epoch.start
                frames.append(raw)
                if frames.count > model.contextEpochs { frames.removeFirst(frames.count-model.contextEpochs) }
                let probabilities = try model.probabilities(frames)
                return SleepPrediction(start: epoch.start, end: epoch.end,
                    probabilities: Dictionary(uniqueKeysWithValues: zip(model.classes, probabilities)), usable: true)
            }
        } catch {
            reset()
            throw error
        }
    }
}

fileprivate struct ContextualModel: Decodable, Sendable {
    enum Kind: String, Decodable, Sendable { case linear, trees }
    let schemaVersion: Int
    let modelKind: Kind
    let modelID: String
    let dataSource: String
    let rawFeatureNames: [String]
    let featureNames: [String]
    let classes: [SleepStage]
    let contextEpochs: Int
    let featureProfile: String?
    let heldOutSubjects: [String]
    let balancedAccuracy: Double
    let biases: [Double]
    let means: [Double]?
    let scales: [Double]?
    let weights: [[Double]]?
    let trees: [Tree]?

    struct Tree: Decodable, Sendable {
        let classIndex: Int
        let left: [Int]
        let right: [Int]
        let features: [Int]
        let values: [Double]

        func validate(dimensions: Int) throws {
            let count = left.count
            guard (0..<4).contains(classIndex), (1...50_000).contains(count),
                  right.count == count, features.count == count, values.count == count,
                  values.allSatisfy({ $0.isFinite && Float($0).isFinite }) else { throw SleepModel.ModelError.invalid }
            var parents = Array(repeating: 0, count: count)
            var depths = Array(repeating: 0, count: count)
            for index in 0..<count {
                if left[index] == -1 {
                    guard right[index] == -1, features[index] == -1 else { throw SleepModel.ModelError.invalid }
                } else {
                    guard (0..<dimensions).contains(features[index]), left[index] > index, right[index] > index,
                          left[index] < count, right[index] < count, left[index] != right[index] else { throw SleepModel.ModelError.invalid }
                    for child in [left[index], right[index]] {
                        parents[child] += 1
                        depths[child] = depths[index] + 1
                        guard depths[child] <= 32 else { throw SleepModel.ModelError.invalid }
                    }
                }
            }
            guard parents[0] == 0, parents.dropFirst().allSatisfy({ $0 == 1 }) else { throw SleepModel.ModelError.invalid }
        }
    }

    func validate() throws {
        let raw = rawFeatureNames
        guard raw == AudioFeatureExtractor.featureNames || (raw == AudioFeatureExtractor.spectralFeatureNames && featureProfile == "spectral-v4") else {
            throw SleepModel.ModelError.invalid
        }
        let expected = contextEpochs == 1 ? raw : raw + ["past_mean_", "past_std_", "past_delta_"].flatMap { prefix in raw.map { prefix + $0 } }
        let dim = expected.count
        guard schemaVersion == 2, contextEpochs == 1 || contextEpochs == 10,
              !modelID.isEmpty, !dataSource.isEmpty, !heldOutSubjects.isEmpty,
              featureNames == expected,
              classes.count == 4, Set(classes) == Set(SleepStage.allCases),
              balancedAccuracy.isFinite, (0...1).contains(balancedAccuracy),
              biases.count == 4, biases.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
        switch modelKind {
        case .linear:
            guard let means, let scales, let weights,
                  means.count == dim, scales.count == dim, weights.count == 4,
                  weights.allSatisfy({ $0.count == dim }),
                  (means + scales + weights.flatMap { $0 }).allSatisfy(\.isFinite),
                  scales.allSatisfy({ $0 > 0 }) else { throw SleepModel.ModelError.invalid }
        case .trees:
            guard let trees, !trees.isEmpty, trees.count <= 4_096,
                  trees.reduce(0, { $0 + $1.left.count }) <= 50_000,
                  biases.allSatisfy({ Float($0).isFinite }) else { throw SleepModel.ModelError.invalid }
            for tree in trees { try tree.validate(dimensions: dim) }
        }
    }

    func probabilities(_ features: [Double]) throws -> [Double] {
        guard features.count == featureNames.count, features.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
        let logits: [Double]
        switch modelKind {
        case .linear:
            guard let means, let scales, let weights else { throw SleepModel.ModelError.invalid }
            let normalized = zip(zip(features, means), scales).map { ($0.0.0 - $0.0.1) / $0.1 }
            logits = zip(weights, biases).map { row, bias in bias + zip(row, normalized).map(*).reduce(0, +) }
        case .trees:
            guard let trees else { throw SleepModel.ModelError.invalid }
            let inputs = features.map(Float.init)
            guard inputs.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
            var margins = biases.map(Float.init)
            for tree in trees {
                var index = 0
                while tree.left[index] != -1 {
                    index = inputs[tree.features[index]] < Float(tree.values[index]) ? tree.left[index] : tree.right[index]
                }
                margins[tree.classIndex] += Float(tree.values[index])
            }
            logits = margins.map(Double.init)
        }
        guard logits.allSatisfy(\.isFinite), let maximum = logits.max() else { throw SleepModel.ModelError.invalid }
        let values = logits.map { exp($0 - maximum) }
        let total = values.reduce(0, +)
        return values.map { $0 / total }
    }
}
