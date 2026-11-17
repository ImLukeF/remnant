import Foundation

struct AcousticSleepModel: Decodable, Sendable {
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
    let encoder: AcousticEncoder
    let recurrent: GRULayer

    func validate() throws {
        try encoder.validate()
        let names = ["mean", "max"].flatMap { kind in (0..<encoder.channels).map { "embedding_\(kind)_\($0)" } }
        guard schemaVersion == 4, modelKind == "conv-gru", !modelID.isEmpty, !dataSource.isEmpty,
              featureProfile == "mel-sequence-v6", rawFeatureNames == AudioFeatureExtractor.acousticFrameFeatureNames,
              featureNames == names, classes.count == 4, Set(classes) == Set(SleepStage.allCases),
              !heldOutSubjects.isEmpty, balancedAccuracy.isFinite, (0...1).contains(balancedAccuracy),
              [1,10,20].contains(contextEpochs) else { throw SleepModel.ModelError.invalid }
        try recurrent.validate(dimensions: encoder.channels*2)
    }
}

struct AcousticEncoder: Decodable, Sendable {
    let channels: Int
    let means: [Double]
    let scales: [Double]
    let normalizedClip: Double
    let weights1: [[[Double]]]
    let biases1: [Double]
    let weights2: [[[Double]]]
    let biases2: [Double]
    let embeddingMeans: [Double]
    let embeddingScales: [Double]
    let embeddingClip: Double

    func validate() throws {
        func kernel(_ values: [[[Double]]], inputs: Int, length: Int) -> Bool {
            values.count == channels && values.allSatisfy { row in
                row.count == inputs && row.allSatisfy { $0.count == length && $0.allSatisfy(\.isFinite) }
            }
        }
        guard (1...16).contains(channels), means.count == 25, scales.count == 25,
              means.allSatisfy(\.isFinite), scales.allSatisfy({ $0.isFinite && $0 > 0 }),
              normalizedClip.isFinite, (1...16).contains(normalizedClip),
              biases1.count == channels, biases2.count == channels, (biases1+biases2).allSatisfy(\.isFinite),
              kernel(weights1, inputs: 25, length: 5), kernel(weights2, inputs: channels, length: 17),
              embeddingMeans.count == channels*2, embeddingScales.count == channels*2,
              embeddingMeans.allSatisfy(\.isFinite), embeddingScales.allSatisfy({ $0.isFinite && $0 > 0 }),
              embeddingClip.isFinite, (1...16).contains(embeddingClip) else { throw SleepModel.ModelError.invalid }
    }

    private func convolution(_ input: [[Double]], weights: [[[Double]]], biases: [Double], length: Int) throws -> [[Double]] {
        var result = Array(repeating: Array(repeating: 0.0, count: channels), count: input.count)
        for time in input.indices {
            for output in 0..<channels {
                var value = 0.0
                for channel in input[0].indices {
                    for offset in 0..<length {
                        let position = time+offset-length/2
                        if position >= 0 && position < input.count { value += weights[output][channel][offset]*input[position][channel] }
                    }
                }
                value += biases[output]
                guard value.isFinite else { throw SleepModel.ModelError.invalid }
                result[time][output] = max(0,value)
            }
        }
        return result
    }

    func encode(_ frames: [Float]) throws -> [Double] {
        guard frames.count == 240*25, frames.allSatisfy(\.isFinite) else { throw SleepModel.ModelError.invalid }
        var input = Array(repeating: Array(repeating: 0.0, count: 25), count: 240)
        for time in 0..<240 {
            for channel in 0..<25 {
                let normalized = (Double(frames[time*25+channel])-means[channel])/scales[channel]
                guard normalized.isFinite else { throw SleepModel.ModelError.invalid }
                input[time][channel] = max(-normalizedClip,min(normalizedClip,normalized))
            }
        }
        let first = try convolution(input, weights: weights1, biases: biases1, length: 5)
        var pooled = Array(repeating: Array(repeating: 0.0, count: channels), count: 120)
        for time in 0..<120 {
            for channel in 0..<channels { pooled[time][channel] = (first[2*time][channel]+first[2*time+1][channel])/2 }
        }
        let second = try convolution(pooled, weights: weights2, biases: biases2, length: 17)
        var embedding = Array(repeating: 0.0, count: channels*2)
        for channel in 0..<channels {
            var total = 0.0, maximum = 0.0
            for time in 0..<120 {
                total += second[time][channel]
                maximum = max(maximum,second[time][channel])
            }
            embedding[channel] = total/120
            embedding[channels+channel] = maximum
        }
        return try embedding.indices.map { index in
            let normalized = (embedding[index]-embeddingMeans[index])/embeddingScales[index]
            guard normalized.isFinite else { throw SleepModel.ModelError.invalid }
            return max(-embeddingClip,min(embeddingClip,normalized))
        }
    }
}
