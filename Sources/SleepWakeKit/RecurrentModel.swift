import Foundation

/// Bounded, unidirectional GRU. Its JSON parameters never authorize changing an alarm.
struct RecurrentSleepModel: Decodable, Sendable {
    let schemaVersion: Int
    let modelKind: String
    let modelID: String
    let dataSource: String
    let featureProfile: String
    let rawFeatureNames: [String]
    let featureNames: [String]
    let classes: [SleepStage]
    let heldOutSubjects: [String]
    let balancedAccuracy: Double
    let contextEpochs: Int
    let hiddenSize: Int
    let normalizedClip: Double
    let means: [Double]
    let scales: [Double]
    let inputWeights: [[Double]]
    let recurrentWeights: [[Double]]
    let inputBiases: [Double]
    let recurrentBiases: [Double]
    let outputWeights: [[Double]]
    let outputBiases: [Double]

    var layer: GRULayer {
        GRULayer(hiddenSize: hiddenSize, inputWeights: inputWeights, recurrentWeights: recurrentWeights,
                 inputBiases: inputBiases, recurrentBiases: recurrentBiases, outputWeights: outputWeights, outputBiases: outputBiases)
    }

    func validate() throws {
        let dimensions = AudioFeatureExtractor.spectralFeatureNames.count
        guard schemaVersion == 3, modelKind == "gru", !modelID.isEmpty, !dataSource.isEmpty,
              featureProfile == "spectral-v4", rawFeatureNames == AudioFeatureExtractor.spectralFeatureNames,
              featureNames == rawFeatureNames, classes.count == 4, Set(classes) == Set(SleepStage.allCases),
              !heldOutSubjects.isEmpty, balancedAccuracy.isFinite, (0...1).contains(balancedAccuracy),
              [1, 10, 20].contains(contextEpochs),
              normalizedClip.isFinite, (1...16).contains(normalizedClip),
              means.count == dimensions, scales.count == dimensions,
              means.allSatisfy(\.isFinite), scales.allSatisfy({ $0.isFinite && $0 > 0 }) else { throw SleepModel.ModelError.invalid }
        try layer.validate(dimensions: dimensions)
    }

    func probabilities(_ frames: [[Double]]) throws -> [Double] {
        guard !frames.isEmpty, frames.count <= contextEpochs else { throw SleepModel.ModelError.invalid }
        let normalized = try frames.map { raw in
            guard raw.count == means.count else { throw SleepModel.ModelError.invalid }
            return try raw.indices.map { index in
                let value = (raw[index]-means[index])/scales[index]
                guard value.isFinite else { throw SleepModel.ModelError.invalid }
                return max(-normalizedClip, min(normalizedClip, value))
            }
        }
        return try layer.probabilities(normalized)
    }
}

/// Shared recurrent equations for raw-observation and learned-embedding models.
struct GRULayer: Decodable, Sendable {
    let hiddenSize: Int
    let inputWeights: [[Double]]
    let recurrentWeights: [[Double]]
    let inputBiases: [Double]
    let recurrentBiases: [Double]
    let outputWeights: [[Double]]
    let outputBiases: [Double]

    func validate(dimensions: Int) throws {
        func matrix(_ values: [[Double]], rows: Int, columns: Int) -> Bool {
            values.count == rows && values.allSatisfy { $0.count == columns && $0.allSatisfy(\.isFinite) }
        }
        guard (1...64).contains(hiddenSize), inputBiases.count == 3*hiddenSize,
              recurrentBiases.count == 3*hiddenSize, outputBiases.count == 4,
              (inputBiases+recurrentBiases+outputBiases).allSatisfy(\.isFinite),
              matrix(inputWeights, rows: 3*hiddenSize, columns: dimensions),
              matrix(recurrentWeights, rows: 3*hiddenSize, columns: hiddenSize),
              matrix(outputWeights, rows: 4, columns: hiddenSize) else { throw SleepModel.ModelError.invalid }
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 { return 1/(1+exp(-value)) }
        let exponential = exp(value)
        return exponential/(1+exponential)
    }

    private static func affine(_ weights: [[Double]], _ values: [Double], _ biases: [Double]) -> [Double] {
        zip(weights, biases).map { row, bias in
            var total = 0.0
            for index in values.indices { total += row[index]*values[index] }
            return bias+total
        }
    }

    func probabilities(_ frames: [[Double]]) throws -> [Double] {
        guard !frames.isEmpty, let dimensions = inputWeights.first?.count else { throw SleepModel.ModelError.invalid }
        var hidden = Array(repeating: 0.0, count: hiddenSize)
        for input in frames {
            guard input.count == dimensions, input.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
            let incoming = Self.affine(inputWeights, input, inputBiases)
            let recurrent = Self.affine(recurrentWeights, hidden, recurrentBiases)
            guard (incoming+recurrent).allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
            var next = Array(repeating: 0.0, count: hiddenSize)
            for index in 0..<hiddenSize {
                let reset = Self.sigmoid(incoming[index]+recurrent[index])
                let update = Self.sigmoid(incoming[hiddenSize+index]+recurrent[hiddenSize+index])
                let candidate = tanh(incoming[2*hiddenSize+index]+reset*recurrent[2*hiddenSize+index])
                next[index] = (1-update)*candidate+update*hidden[index]
            }
            hidden = next
        }
        let logits = Self.affine(outputWeights, hidden, outputBiases)
        guard logits.allSatisfy(\.isFinite), let maximum = logits.max() else { throw SleepModel.ModelError.invalid }
        let values = logits.map { exp($0-maximum) }
        let total = values.reduce(0,+)
        return values.map { $0/total }
    }
}
