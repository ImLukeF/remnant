import Foundation

public enum SleepStage: String, Codable, CaseIterable, Sendable {
    case awake, light, deep, rem
}

public struct SleepPrediction: Codable, Equatable, Sendable {
    public let start: Date
    public let end: Date
    public let probabilities: [SleepStage: Double]
    public let usable: Bool

    public var stage: SleepStage? { probabilities.max { $0.value < $1.value }?.key }
    public var confidence: Double { probabilities.values.max() ?? 0 }

    public init(start: Date, end: Date, probabilities: [SleepStage: Double], usable: Bool) {
        self.start = start
        self.end = end
        self.probabilities = probabilities
        self.usable = usable
    }
}

/// A small, auditable baseline trained on labelled audio features. No weights ship with the library.
/// Imported metadata does not grant a model permission to control an alarm.
public struct LinearSleepModel: Codable, Sendable {
    public let schemaVersion: Int
    public let featureNames: [String]
    public let modelID: String
    public let dataSource: String
    public let classes: [SleepStage]
    public let means: [Double]
    public let scales: [Double]
    public let weights: [[Double]]
    public let biases: [Double]
    public let heldOutSubjects: [String]
    public let balancedAccuracy: Double

    public init(data: Data) throws {
        guard data.count <= 512_000 else { throw ModelError.invalid }
        self = try JSONDecoder().decode(Self.self, from: data)
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        featureNames = try values.decode([String].self, forKey: .featureNames)
        modelID = try values.decode(String.self, forKey: .modelID)
        dataSource = try values.decode(String.self, forKey: .dataSource)
        classes = try values.decode([SleepStage].self, forKey: .classes)
        means = try values.decode([Double].self, forKey: .means)
        scales = try values.decode([Double].self, forKey: .scales)
        weights = try values.decode([[Double]].self, forKey: .weights)
        biases = try values.decode([Double].self, forKey: .biases)
        heldOutSubjects = try values.decode([String].self, forKey: .heldOutSubjects)
        balancedAccuracy = try values.decode(Double.self, forKey: .balancedAccuracy)
        let count = AudioFeatureExtractor.featureNames.count
        guard schemaVersion == 1, featureNames == AudioFeatureExtractor.featureNames,
              !modelID.isEmpty, !dataSource.isEmpty,
              classes.count == 4, Set(classes) == Set(SleepStage.allCases),
              means.count == count, scales.count == count,
              weights.count == classes.count, biases.count == classes.count,
              weights.allSatisfy({ $0.count == count }),
              (means + scales + weights.flatMap { $0 } + biases).allSatisfy(\.isFinite),
              scales.allSatisfy({ $0 > 0 }),
              balancedAccuracy.isFinite, (0...1).contains(balancedAccuracy),
              !heldOutSubjects.isEmpty else { throw ModelError.invalid }
    }

    public func predict(_ epoch: AudioEpoch) throws -> SleepPrediction {
        guard epoch.features.count == means.count, epoch.features.allSatisfy(\.isFinite),
              epoch.end > epoch.start else { throw ModelError.invalid }
        let normalized = zip(zip(epoch.features, means), scales).map { ($0.0.0 - $0.0.1) / $0.1 }
        let logits = zip(weights, biases).map { row, bias in
            zip(row, normalized).map(*).reduce(bias, +)
        }
        guard logits.allSatisfy(\.isFinite), let maximum = logits.max() else { throw ModelError.invalid }
        let values = logits.map { exp($0 - maximum) }
        let total = values.reduce(0, +)
        return SleepPrediction(start: epoch.start, end: epoch.end,
                               probabilities: Dictionary(uniqueKeysWithValues: zip(classes, values.map { $0 / total })),
                               usable: epoch.usable)
    }

    public enum ModelError: Error { case invalid }
}
