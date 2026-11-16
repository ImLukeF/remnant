import Foundation
import Testing
@testable import SleepWakeKit

@Test func audioFrameContinuityRejectsDroppedBuffersAndOverflow() {
    var timeline = AudioFrameContinuity()
    let outcomes = [timeline.accept(sampleTime: 0, frameCount: 4096),
                    timeline.accept(sampleTime: 4096, frameCount: 4096),
                    timeline.accept(sampleTime: 12288, frameCount: 4096),
                    timeline.accept(sampleTime: 12288, frameCount: 4096),
                    timeline.accept(sampleTime: Int64.max - 1, frameCount: 4096),
                    timeline.accept(sampleTime: 0, frameCount: 0)]
    #expect(outcomes == [true, true, false, true, false, false])
}

@Test func streamingFeaturesDoNotDependOnBufferBoundaries() throws {
    let start = Date(timeIntervalSince1970: 1_000)
    let pcm = (0..<(8_000 * 30)).map { Float($0.isMultiple(of: 2) ? 0.5 : -0.5) }
    var whole = try AudioFeatureExtractor(sampleRate: 8_000, start: start)
    var streamed = try AudioFeatureExtractor(sampleRate: 8_000, start: start)
    let expected = whole.append(pcm)
    var actual: [AudioEpoch] = []
    for offset in stride(from: 0, to: pcm.count, by: 4_096) {
        actual += streamed.append(Array(pcm[offset..<min(offset + 4_096, pcm.count)]))
    }
    #expect(actual == expected)
    let epoch = try #require(actual.first)
    #expect(epoch.start == start && epoch.end == start.addingTimeInterval(30))
    #expect(abs(epoch.features[0] - (-6.020599913)) < 1e-8)
    #expect(epoch.features[1] < 1e-10)
    #expect(epoch.features[4] > 0.999)
    #expect(epoch.usable)
}

@Test func partialMissingClippedAndInvalidAudioCannotBecomeUsableEpochs() throws {
    var extractor = try AudioFeatureExtractor(sampleRate: 8_000, start: .distantPast)
    #expect(extractor.append(Array(repeating: 0, count: 8_000 * 29)).isEmpty)
    #expect(extractor.append(Array(repeating: 0, count: 8_000)).first?.usable == false)
    #expect(extractor.append(Array(repeating: 1, count: 8_000 * 30)).first?.usable == false)
    var invalid = Array(repeating: Float(0.1), count: 8_000 * 30)
    invalid[1] = .nan
    #expect(extractor.append(invalid).first?.usable == false)
}

@Test func stopKeepsSafetySequenceAndFinishIsScopedAndIdempotent() throws {
    let deadline = Date(timeIntervalSince1970: 10_000)
    let plan = try WakePlan(upBy: deadline)
    #expect(plan.attempts.map { $0.date.timeIntervalSince(deadline) } == [-600, -300, 0])
    #expect(plan.earliestWake == deadline.addingTimeInterval(-1_800))
    var session = WakeSession(plan: plan)
    session.stopped(UUID())
    #expect(session.stoppedIDs.isEmpty)
    session.stopped(plan.attempts[0].id)
    #expect(session.plan.attempts == plan.attempts)
    #expect(session.finishedAt == nil)
    #expect(session.finish(at: deadline) == plan.attempts.map(\.id))
    #expect(session.finish(at: deadline).isEmpty)
    #expect(throws: WakePlan.PlanError.self) { try WakePlan(upBy: deadline, attemptCount: 0) }
    #expect(throws: WakePlan.PlanError.self) {
        try JSONDecoder().decode(WakePlan.self, from: Data("""
        {"id":"\(UUID())","attempts":[],"earliestWake":0}
        """.utf8))
    }
}

@Test func onlyApprovedFreshContiguousLightPredictionsCanSuggestOneEarlyAlarm() throws {
    let deadline = Date(timeIntervalSince1970: 10_000)
    let now = deadline.addingTimeInterval(-900)
    let plan = try WakePlan(upBy: deadline)
    func light(ending end: Date) -> SleepPrediction {
        SleepPrediction(start: end.addingTimeInterval(-30), end: end,
                        probabilities: [.awake: 0.03, .light: 0.91, .deep: 0.03, .rem: 0.03], usable: true)
    }
    for badEnd in [now.addingTimeInterval(-300), now.addingTimeInterval(30)] {
        var session = WakeSession(plan: plan)
        #expect(session.consider(light(ending: badEnd), now: now, modelApprovedForAlarmTiming: true) == nil)
        #expect(session.consider(light(ending: now), now: now, modelApprovedForAlarmTiming: true) == nil)
    }
    var unapproved = WakeSession(plan: plan)
    #expect(unapproved.consider(light(ending: now.addingTimeInterval(-30)), now: now) == nil)
    #expect(unapproved.consider(light(ending: now), now: now) == nil)
    var session = WakeSession(plan: plan)
    #expect(session.consider(light(ending: now.addingTimeInterval(-30)), now: now, modelApprovedForAlarmTiming: true) == nil)
    let suggested = session.consider(light(ending: now), now: now, modelApprovedForAlarmTiming: true)
    let attempt = try #require(suggested)
    #expect(attempt.date == now.addingTimeInterval(2))
    #expect(session.plan.attempts == plan.attempts)
    #expect(session.consider(light(ending: now.addingTimeInterval(30)), now: now.addingTimeInterval(30),
                             modelApprovedForAlarmTiming: true) == nil)
}

@Test func linearModelChecksSchemaAndComputesStableProbabilities() throws {
    func data(names: [String], bias: Double = 1_000) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "schemaVersion": 1, "featureNames": names, "modelID": "math-fixture", "dataSource": "synthetic-test",
            "classes": ["awake", "light", "deep", "rem"], "means": Array(repeating: 0, count: 7),
            "scales": Array(repeating: 1, count: 7), "weights": Array(repeating: Array(repeating: 0, count: 7), count: 4),
            "biases": [0, bias, 0, 0], "heldOutSubjects": ["fixture"], "balancedAccuracy": 0,
        ])
    }
    let model = try LinearSleepModel(data: data(names: AudioFeatureExtractor.featureNames))
    let prediction = try model.predict(AudioEpoch(start: .distantPast, end: .distantPast.addingTimeInterval(30),
                                                 features: Array(repeating: 0, count: 7), usable: true))
    #expect(prediction.stage == .light && prediction.confidence == 1)
    #expect(prediction.probabilities.values.reduce(0, +) == 1)
    var compatible = try SleepModel(data: data(names: AudioFeatureExtractor.featureNames)).makePredictor()
    #expect(try compatible.predict(AudioEpoch(start: .distantPast, end: .distantPast.addingTimeInterval(30),
        features: Array(repeating: 0, count: 7), usable: true)).stage == .light)
    #expect(throws: LinearSleepModel.ModelError.self) {
        try LinearSleepModel(data: data(names: ["wrong"]))
    }
}

@Test func researchModelContextUsesOnlyThePastAndResetsOnGapsOrUnusableAudio() throws {
    let raw = AudioFeatureExtractor.featureNames
    var weights = Array(repeating: Array(repeating: 0.0, count: 28), count: 4)
    weights[1][7] = 1
    let data = try JSONSerialization.data(withJSONObject: [
        "schemaVersion": 2, "modelKind": "linear", "modelID": "context-fixture", "dataSource": "synthetic",
        "rawFeatureNames": raw, "featureNames": raw + ["past_mean_", "past_std_", "past_delta_"].flatMap { prefix in raw.map { prefix + $0 } },
        "classes": ["awake", "light", "deep", "rem"], "contextEpochs": 10,
        "heldOutSubjects": ["fixture"], "balancedAccuracy": 0, "biases": [0, 0, 0, 0],
        "means": Array(repeating: 0, count: 28), "scales": Array(repeating: 1, count: 28), "weights": weights,
    ])
    var predictor = try SleepModel(data: data).makePredictor()
    func epoch(_ start: Double, _ value: Double, usable: Bool = true) -> AudioEpoch {
        AudioEpoch(start: Date(timeIntervalSince1970: start), end: Date(timeIntervalSince1970: start + 30),
                   features: Array(repeating: value, count: 7), usable: usable)
    }
    #expect(try predictor.predict(epoch(0, 0)).probabilities[.light] == 0.25)
    let second = try predictor.predict(epoch(30, 2))
    #expect(abs((second.probabilities[.light] ?? 0) - exp(1) / (exp(1) + 3)) < 1e-12)
    let afterGap = try predictor.predict(epoch(300, 2))
    #expect(abs((afterGap.probabilities[.light] ?? 0) - exp(2) / (exp(2) + 3)) < 1e-12)
    let invalid = try predictor.predict(epoch(330, 0, usable: false))
    #expect(!invalid.usable && invalid.probabilities.isEmpty)
    #expect(try predictor.predict(epoch(360, 2)).probabilities == afterGap.probabilities)
    predictor.reset()
    #expect(try predictor.predict(epoch(390, 2)).probabilities == afterGap.probabilities)
}

@Test func researchTreeImportValidatesGraphsAndUsesStrictFloat32Thresholds() throws {
    func payload(left: [Int] = [1, -1, -1], features: [Int] = [0, -1, -1], values: [Double] = [1, 2, -2], spectral: Bool = false) throws -> Data {
        let raw = spectral ? AudioFeatureExtractor.spectralFeatureNames : AudioFeatureExtractor.featureNames
        var object: [String: Any] = [
            "schemaVersion": 2, "modelKind": "trees", "modelID": "tree-fixture", "dataSource": "synthetic",
            "rawFeatureNames": raw, "featureNames": raw, "classes": ["awake", "light", "deep", "rem"],
            "contextEpochs": 1, "heldOutSubjects": ["fixture"], "balancedAccuracy": 0, "biases": [0, 0, 0, 0],
            "trees": [["classIndex": 1, "left": left, "right": [2, -1, -1], "features": features, "values": values]],
        ]
        if spectral { object["featureProfile"] = "spectral-v4" }
        return try JSONSerialization.data(withJSONObject: object)
    }
    var predictor = try SleepModel(data: payload()).makePredictor()
    func epoch(_ value: Double) -> AudioEpoch {
        AudioEpoch(start: Date(timeIntervalSince1970: 0), end: Date(timeIntervalSince1970: 30), features: Array(repeating: value, count: 7), usable: true)
    }
    #expect(try predictor.predict(epoch(0.5)).probabilities[.light]! > 0.7)
    #expect(try predictor.predict(epoch(1)).probabilities[.light]! < 0.05)
    let upgraded = AudioEpoch(start: Date(timeIntervalSince1970: 0), end: Date(timeIntervalSince1970: 30),
        features: Array(repeating: 0.5, count: AudioFeatureExtractor.spectralFeatureNames.count), usable: true)
    #expect(try predictor.predict(upgraded).probabilities[.light]! > 0.7)
    var spectralPredictor = try SleepModel(data: payload(features: [7, -1, -1], values: [0.5000002, 2, -2], spectral: true)).makePredictor()
    let spectralInput = AudioEpoch(start: upgraded.start, end: upgraded.end,
        features: Array(repeating: 0.5000004, count: AudioFeatureExtractor.spectralFeatureNames.count), usable: true)
    #expect(try spectralPredictor.predict(spectralInput).probabilities[.light]! > 0.7)
    #expect(throws: SleepModel.ModelError.self) { try spectralPredictor.predict(epoch(0.5)) }
    for bad in [try payload(left: [0, -1, -1]), try payload(features: [7, -1, -1]), try payload(values: [1, 1e100, -2])] {
        #expect(throws: SleepModel.ModelError.self) { try SleepModel(data: bad) }
    }
    #expect(throws: SleepModel.ModelError.self) {
        try SleepModel(data: Data(repeating: 32, count: SleepModel.maximumPayloadBytes + 1))
    }
}

@Test func spectralFeaturesMatchIndependentReferenceAcrossStreamingBuffers() throws {
    let referenceURL = try #require(Bundle.module.url(forResource: "spectral-reference", withExtension: "json"))
    let expected = try JSONDecoder().decode([Double].self, from: Data(contentsOf: referenceURL))
    var extractor = try AudioFeatureExtractor(sampleRate: 8000, start: Date(timeIntervalSince1970: 0), spectral: true)
    var state: UInt32 = 7
    var chunk: [Float] = []
    var epochs: [AudioEpoch] = []
    for index in 0..<(8000 * 30) {
        state = 1664525 &* state &+ 1013904223
        let noise = Float(Int(state >> 16) - 32768) / 262144
        chunk.append(noise + (index / 4 % 2 == 0 ? 0.125 : -0.125))
        if chunk.count == 3079 { epochs += extractor.append(chunk); chunk.removeAll(keepingCapacity: true) }
    }
    epochs += extractor.append(chunk)
    let result = try #require(epochs.first)
    #expect(epochs.count == 1 && result.usable)
    #expect(result.features.count == AudioFeatureExtractor.spectralFeatureNames.count)
    for (actual, expected) in zip(result.features, expected) { #expect(abs(actual - expected) < 2e-6) }
    let dc = extractor.append(Array(repeating: 0.25, count: 8000 * 30))
    #expect(dc.count == 1 && !dc[0].usable)
}

@Test func finishingTodayDoesNotSkipTomorrowsRepeatingAlarm() {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(secondsFromGMT: 0)!
    let now = Date(timeIntervalSince1970: 23_000)
    let ringing = UUID(), later = UUID(), tomorrow = UUID(), oneOff = UUID(), skipped = UUID()
    let members = [
        ScheduledWakeMember(id: ringing, nextDate: now.addingTimeInterval(86_400), repeats: true, isActive: true, isSkipping: false),
        ScheduledWakeMember(id: later, nextDate: now.addingTimeInterval(300), repeats: true, isActive: false, isSkipping: false),
        ScheduledWakeMember(id: tomorrow, nextDate: now.addingTimeInterval(86_400), repeats: true, isActive: false, isSkipping: false),
        ScheduledWakeMember(id: oneOff, nextDate: now.addingTimeInterval(600), repeats: false, isActive: false, isSkipping: false),
        ScheduledWakeMember(id: skipped, nextDate: now.addingTimeInterval(900), repeats: true, isActive: false, isSkipping: true),
    ]
    #expect(WakeCompletion.actions(for: members, now: now, calendar: calendar)
            == [.stop(ringing), .skipNext(later), .cancel(oneOff)])
}
