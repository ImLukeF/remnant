import Foundation

/// Compile with Sources/SleepWakeKit/*.swift; no app or microphone session is started.
@main
struct VerifyNativeExport {
    struct Observation: Decodable {
        let subject: String
        let startSeconds: Double
        let features: [Double]
        let usable: Bool
        let acousticFrames: [Float]?
        let expectedProbabilities: [String: Double]?
    }

    enum Failure: Error { case arguments, invalidReference, mismatch }

    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw Failure.arguments }
        let modelData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let referenceData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
        let model = try SleepModel(data: modelData)
        let rows = try JSONDecoder().decode([Observation].self, from: referenceData)
        guard !rows.isEmpty else { throw Failure.invalidReference }
        var predictor = model.makePredictor()
        var previousSubject: String?
        var subjects = Set<String>()
        var maximumDifference = 0.0
        var comparisons = 0
        let stageNames = Set(SleepStage.allCases.map(\.rawValue))
        for row in rows {
            guard !row.subject.isEmpty else { throw Failure.invalidReference }
            if row.subject != previousSubject {
                // Returning to a person would silently omit their earlier causal context.
                guard subjects.insert(row.subject).inserted else { throw Failure.invalidReference }
                predictor.reset()
                previousSubject = row.subject
            }
            let start = Date(timeIntervalSince1970: row.startSeconds)
            let epoch = AudioEpoch(start: start, end: start.addingTimeInterval(30),
                features: row.features, usable: row.usable, acousticFrames: row.acousticFrames)
            let actual = try predictor.predict(epoch)
            if row.usable {
                guard let expected = row.expectedProbabilities,
                      Set(expected.keys) == stageNames,
                      expected.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                      abs(expected.values.reduce(0,+)-1) < 1e-10,
                      actual.usable, actual.probabilities.count == stageNames.count else {
                    throw Failure.invalidReference
                }
                for stage in SleepStage.allCases {
                    guard let value = actual.probabilities[stage], let reference = expected[stage.rawValue] else {
                        throw Failure.invalidReference
                    }
                    maximumDifference = max(maximumDifference, abs(value-reference))
                    comparisons += 1
                }
            } else {
                guard row.expectedProbabilities == nil, !actual.usable, actual.probabilities.isEmpty else {
                    throw Failure.invalidReference
                }
            }
        }
        guard comparisons > 0, maximumDifference <= 2e-10 else { throw Failure.mismatch }
        let receipt: [String: Any] = ["observations": rows.count, "people": subjects.count,
            "probabilityComparisons": comparisons, "maximumProbabilityDifference": maximumDifference,
            "modelBytes": modelData.count, "approvedForAlarmTiming": false]
        let data = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
