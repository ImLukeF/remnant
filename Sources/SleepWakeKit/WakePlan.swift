import Foundation

public struct WakeAttempt: Codable, Equatable, Identifiable, Sendable {
    public let id: UUID
    public let date: Date

    public init(id: UUID = UUID(), date: Date) { self.id = id; self.date = date }
}

public struct ScheduledWakeMember: Sendable {
    public let id: UUID
    public let nextDate: Date
    public let repeats: Bool
    public let isActive: Bool
    public let isSkipping: Bool

    public init(id: UUID, nextDate: Date, repeats: Bool, isActive: Bool, isSkipping: Bool) {
        self.id = id; self.nextDate = nextDate; self.repeats = repeats
        self.isActive = isActive; self.isSkipping = isSkipping
    }
}

public enum WakeCompletionAction: Equatable, Sendable {
    case stop(UUID), cancel(UUID), skipNext(UUID)
}

public enum WakeCompletion {
    /// Members must already be scoped to this plan. Finish today's occurrence without skipping tomorrow.
    public static func actions(for members: [ScheduledWakeMember], now: Date,
                               calendar: Calendar = .current) -> [WakeCompletionAction] {
        var result: [WakeCompletionAction] = []
        for member in members {
            if member.isActive { result.append(.stop(member.id)) }
            guard calendar.isDate(member.nextDate, inSameDayAs: now), !member.isSkipping else { continue }
            if !member.repeats { result.append(.cancel(member.id)) }
            else if member.nextDate > now { result.append(.skipNext(member.id)) }
        }
        return result
    }
}

/// One occurrence. The host schedules every safety attempt with the operating system before starting capture.
public struct WakePlan: Codable, Equatable, Sendable {
    public let id: UUID
    public let attempts: [WakeAttempt]
    public let earliestWake: Date
    public var upBy: Date { attempts.last!.date }
    public var firstAttempt: Date { attempts.first!.date }

    public init(id: UUID = UUID(), upBy: Date, attemptCount: Int = 3,
                spacing: TimeInterval = 300, earlyAllowance: TimeInterval = 1_800) throws {
        guard upBy.timeIntervalSinceReferenceDate.isFinite,
              (1...12).contains(attemptCount), spacing.isFinite, spacing >= 60, spacing <= 1_800,
              earlyAllowance.isFinite, (0...7_200).contains(earlyAllowance) else { throw PlanError.invalid }
        let first = upBy.addingTimeInterval(-Double(attemptCount - 1) * spacing)
        self.id = id
        self.earliestWake = min(first, upBy.addingTimeInterval(-earlyAllowance))
        self.attempts = (0..<attemptCount).map {
            WakeAttempt(date: first.addingTimeInterval(Double($0) * spacing))
        }
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(UUID.self, forKey: .id)
        attempts = try values.decode([WakeAttempt].self, forKey: .attempts)
        earliestWake = try values.decode(Date.self, forKey: .earliestWake)
        guard !attempts.isEmpty, attempts.count <= 12, Set(attempts.map(\.id)).count == attempts.count,
              earliestWake.timeIntervalSinceReferenceDate.isFinite,
              attempts.allSatisfy({ $0.date.timeIntervalSinceReferenceDate.isFinite }),
              zip(attempts, attempts.dropFirst()).allSatisfy({ $0.date < $1.date }),
              earliestWake <= attempts[0].date else { throw PlanError.invalid }
    }

    public enum PlanError: Error { case invalid }
}

/// Stopping an attempt never removes the safety sequence. Finishing explicitly returns the IDs to cancel.
public struct WakeSession: Codable, Equatable, Sendable {
    public let plan: WakePlan
    public private(set) var stoppedIDs: Set<UUID> = []
    public private(set) var finishedAt: Date?
    public private(set) var earlyAttempt: WakeAttempt?
    private var previousLightEnd: Date?

    public init(plan: WakePlan) { self.plan = plan }

    public mutating func stopped(_ id: UUID) {
        guard plan.attempts.contains(where: { $0.id == id }) || earlyAttempt?.id == id else { return }
        stoppedIDs.insert(id)
    }

    public mutating func finish(at now: Date) -> [UUID] {
        guard finishedAt == nil else { return [] }
        finishedAt = now
        previousLightEnd = nil
        return plan.attempts.map(\.id) + [earlyAttempt?.id].compactMap { $0 }
    }

    /// The app must independently approve a model for alarm timing after validation.
    /// Two contiguous current light predictions are required; offline/future/stale estimates cannot ring.
    public mutating func consider(_ prediction: SleepPrediction, now: Date,
                                  modelApprovedForAlarmTiming: Bool = false) -> WakeAttempt? {
        let probabilities = Array(prediction.probabilities.values)
        let valid = prediction.start < prediction.end && prediction.end <= now
            && now.timeIntervalSince(prediction.end) <= 60
            && (29...31).contains(prediction.end.timeIntervalSince(prediction.start))
            && probabilities.count == 4 && probabilities.allSatisfy { $0.isFinite && (0...1).contains($0) }
            && abs(probabilities.reduce(0, +) - 1) < 0.001
        guard finishedAt == nil, earlyAttempt == nil, stoppedIDs.isEmpty,
              now >= plan.earliestWake, now < plan.firstAttempt,
              modelApprovedForAlarmTiming, prediction.usable, valid,
              prediction.stage == .light, prediction.confidence >= 0.85 else {
            previousLightEnd = nil
            return nil
        }
        defer { previousLightEnd = prediction.end }
        guard let previousLightEnd,
              abs(prediction.start.timeIntervalSince(previousLightEnd)) <= 0.5,
              prediction.end > previousLightEnd else { return nil }
        let attempt = WakeAttempt(date: now.addingTimeInterval(2))
        guard attempt.date < plan.firstAttempt else { return nil }
        earlyAttempt = attempt
        return attempt
    }
}
