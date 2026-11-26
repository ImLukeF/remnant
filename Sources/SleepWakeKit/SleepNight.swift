import Foundation

/// Phone observations, personal feedback and Health references retain separate provenance.
public struct SleepNight: Codable, Equatable, Identifiable, Sendable {
    public enum EndReason: String, Codable, Sendable { case finished, interrupted, appTerminated, cancelled, storageFailure, limit }
    public enum MarkerKind: String, Codable, CaseIterable, Sendable { case awake, tryingToSleep, tvOff }
    public struct Marker: Codable, Equatable, Identifiable, Sendable {
        public var id = UUID()
        public var date: Date
        public var kind: MarkerKind
        public init(date: Date, kind: MarkerKind) { self.date = date; self.kind = kind }
    }
    public struct Feedback: Codable, Equatable, Sendable {
        public var sleepQuality: Int?
        public var wakeDifficulty: Int?
        public var fellAsleepAt: Date?
        public var wokeAt: Date?
        public var notes = ""
        public init() {}
    }
    public struct HealthReference: Codable, Equatable, Sendable {
        public var start: Date
        public var end: Date
        public var category: Int
        public var source: String
        public var sourceBundleID: String
        public var device: String?
        public init(start: Date, end: Date, category: Int, source: String, sourceBundleID: String, device: String?) {
            self.start = start; self.end = end; self.category = category; self.source = source
            self.sourceBundleID = sourceBundleID; self.device = device
        }
    }
    /// The night read again after it ended, one stage per observation; nil where the saved sound doesn't cover it.
    public struct Morning: Codable, Equatable, Sendable {
        public var modelID: String
        public var stages: [SleepStage?]
        public init(modelID: String, stages: [SleepStage?]) { self.modelID = modelID; self.stages = stages }
    }
    public struct Preview: Codable, Equatable, Identifiable, Sendable {
        public var id: UUID
        public var start: Date
        public var end: Date
        public var usable: Bool
        public var stage: SleepStage?
        public var confidence: Double?
        public var soundLevel: Double
        public var battery: Double?
    }

    public var schemaVersion = 1
    public var dateEncoding = "secondsSince2001-01-01T00:00:00Z"
    public let id: UUID
    public var started: Date
    public var ended: Date?
    public var endReason: EndReason?
    public var modelID: String
    public var modelSHA256: String?
    public var inputSampleRate: Int?
    public var audioSampleRate: Int?
    public var appBuild: String?
    public var deviceModel: String?
    public var systemVersion: String?
    public var featureProfile = "spectral-v4"
    public var recordsAudio: Bool
    public var expectedTVOff: Date?
    public var markers: [Marker] = []
    public var observations: [Preview] = []
    public var feedback = Feedback()
    public var healthReference: [HealthReference] = []
    public var healthReferenceReadAt: Date?
    public var morning: Morning?
    public var capturedSeconds: Int { observations.count * AudioFeatureExtractor.epochSeconds }
    public var usableSeconds: Int { observations.filter(\.usable).count * AudioFeatureExtractor.epochSeconds }

    public init(id: UUID = UUID(), started: Date, modelID: String, recordsAudio: Bool, tvMinutes: Int) {
        self.id = id; self.started = started; self.modelID = modelID; self.recordsAudio = recordsAudio
        expectedTVOff = (1...180).contains(tvMinutes) ? started.addingTimeInterval(Double(tvMinutes) * 60) : nil
    }
}

public struct SleepNightObservation: Codable, Equatable, Identifiable, Sendable {
    public let id: UUID
    public let epoch: AudioEpoch
    public let prediction: SleepPrediction?
    public let battery: Double?
    public let duringExpectedTV: Bool
    public init(id: UUID = UUID(), epoch: AudioEpoch, prediction: SleepPrediction?, battery: Double?, duringExpectedTV: Bool) {
        self.id = id; self.epoch = epoch; self.prediction = prediction; self.battery = battery
        self.duringExpectedTV = duringExpectedTV
    }
    public var preview: SleepNight.Preview {
        .init(id: id, start: epoch.start, end: epoch.end, usable: epoch.usable,
              stage: prediction?.stage, confidence: prediction?.confidence,
              soundLevel: epoch.features.first ?? -120, battery: battery)
    }
}

/// One owner calls this store. Observations are atomic individual files, never rewritten with the whole night.
public final class SleepNightStore {
    public enum Failure: Error { case invalidNight, activeNight, invalidObservation, tooManyObservations }
    public let directory: URL
    public init(directory: URL) throws {
        self.directory = directory
        try Self.makePrivateDirectory(directory)
    }

    public func folder(_ id: UUID) -> URL { directory.appending(path: id.uuidString, directoryHint: .isDirectory) }
    public func audioURL(_ id: UUID) -> URL { folder(id).appending(path: "audio.caf") }

    public func create(_ night: SleepNight, modelData: Data? = nil, modelAttribution: Data? = nil) throws {
        guard !FileManager.default.fileExists(atPath: folder(night.id).path) else { throw Failure.invalidNight }
        try Self.makePrivateDirectory(folder(night.id))
        try Self.makePrivateDirectory(folder(night.id).appending(path: "observations", directoryHint: .isDirectory))
        if let modelData {
            guard try SleepModel(data: modelData).identifier == night.modelID else { throw Failure.invalidNight }
            try Self.write(modelData, to: folder(night.id).appending(path: "model.json"))
        }
        if let modelAttribution { try Self.write(modelAttribution, to: folder(night.id).appending(path: "model-attribution.txt")) }
        try save(night)
    }

    public func save(_ night: SleepNight) throws {
        guard night.schemaVersion == 1, night.started.timeIntervalSince1970.isFinite else { throw Failure.invalidNight }
        try Self.write(JSONEncoder().encode(night), to: folder(night.id).appending(path: "night.json"))
    }

    public func append(_ observation: SleepNightObservation, to night: inout SleepNight) throws {
        guard night.ended == nil else { throw Failure.activeNight }
        guard night.observations.count < 1_440 else { throw Failure.tooManyObservations }
        let epoch = observation.epoch
        guard (29...31).contains(epoch.end.timeIntervalSince(epoch.start)), epoch.start.timeIntervalSince1970.isFinite,
              epoch.end.timeIntervalSince1970.isFinite, epoch.features.count == 58, epoch.features.allSatisfy(\.isFinite),
              (epoch.acousticFrames == nil || epoch.acousticFrames?.count == 6_000),
              epoch.acousticFrames?.allSatisfy(\.isFinite) ?? true,
              observation.battery.map({ $0.isFinite && (0...1).contains($0) }) ?? true,
              night.observations.last.map({ epoch.start >= $0.end }) ?? true else { throw Failure.invalidObservation }
        let url = folder(night.id).appending(path: "observations/\(observation.id.uuidString).json")
        guard !FileManager.default.fileExists(atPath: url.path) else { throw Failure.invalidObservation }
        try Self.write(JSONEncoder().encode(observation), to: url)
        night.observations.append(observation.preview)
        try save(night)
    }

    public func nights(recoverInterrupted: Bool = false) throws -> [SleepNight] {
        let folders = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        var result: [SleepNight] = []
        for url in folders {
            guard let id = UUID(uuidString: url.lastPathComponent),
                  let data = try? Data(contentsOf: url.appending(path: "night.json")),
                  var night = try? JSONDecoder().decode(SleepNight.self, from: data), night.id == id else { continue }
            if recoverInterrupted, night.ended == nil {
                // Recover an observation written just before a process kill interrupted its manifest update.
                let known = Set(night.observations.map(\.id))
                let files = try FileManager.default.contentsOfDirectory(at: url.appending(path: "observations"), includingPropertiesForKeys: nil)
                let extras = try files.compactMap { file -> SleepNightObservation? in
                    guard let id = UUID(uuidString: file.deletingPathExtension().lastPathComponent), !known.contains(id) else { return nil }
                    let item = try JSONDecoder().decode(SleepNightObservation.self, from: Data(contentsOf: file))
                    guard item.id == id else { throw Failure.invalidObservation }
                    return item
                }.sorted { $0.epoch.start < $1.epoch.start }
                for item in extras {
                    guard night.observations.last.map({ item.epoch.start >= $0.end }) ?? true else { throw Failure.invalidObservation }
                    night.observations.append(item.preview)
                }
                night.ended = night.observations.last?.end ?? night.started
                night.endReason = .appTerminated
                try save(night)
            }
            result.append(night)
        }
        return result.sorted { $0.started > $1.started }
    }

    public func delete(_ night: SleepNight) throws {
        let current = try JSONDecoder().decode(SleepNight.self, from: Data(contentsOf: folder(night.id).appending(path: "night.json")))
        guard night.ended != nil, current.ended != nil else { throw Failure.activeNight }
        try FileManager.default.removeItem(at: folder(night.id))
    }

    public func export(_ night: SleepNight, to destination: URL, includeAudio: Bool) throws {
        guard night.ended != nil else { throw Failure.activeNight }
        let current = try JSONDecoder().decode(SleepNight.self, from: Data(contentsOf: folder(night.id).appending(path: "night.json")))
        guard current == night else { throw Failure.invalidNight }
        var files = [("night.json", folder(night.id).appending(path: "night.json"))]
        let model = folder(night.id).appending(path: "model.json")
        if FileManager.default.fileExists(atPath: model.path) { files.append(("model.json", model)) }
        let attribution = folder(night.id).appending(path: "model-attribution.txt")
        if FileManager.default.fileExists(atPath: attribution.path) { files.append(("model-attribution.txt", attribution)) }
        files += night.observations.map { ("observations/\($0.id.uuidString).json", folder(night.id).appending(path: "observations/\($0.id.uuidString).json")) }
        if includeAudio, night.recordsAudio, FileManager.default.fileExists(atPath: audioURL(night.id).path) {
            files.append(("audio.caf", audioURL(night.id)))
        }
        try SleepNightArchive.write(files, to: destination)
    }

    public static func makePrivateDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        #if canImport(Darwin)
        var resource = URLResourceValues(); resource.isExcludedFromBackup = true
        var url = url; try url.setResourceValues(resource)
        #endif
    }

    private static func write(_ data: Data, to url: URL) throws {
        #if os(iOS)
        try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        #else
        try data.write(to: url, options: .atomic)
        #endif
    }
}
