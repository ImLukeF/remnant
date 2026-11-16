import Foundation
import Testing
@testable import SleepWakeKit

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
    #expect(throws: LinearSleepModel.ModelError.self) {
        try LinearSleepModel(data: data(names: ["wrong"]))
    }
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
