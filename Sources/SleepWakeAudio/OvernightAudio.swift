#if os(iOS)
import AVFAudio
import Foundation
import SleepWakeKit

/// Explicitly started audio observation. Stores features only; never writes recordings or uses the network.
@MainActor
public final class OvernightAudio {
    public enum Event: Sendable {
        case epoch(AudioEpoch)
        case interrupted
        case stopped
    }
    public enum Failure: Error { case permissionDenied, invalidInput, alreadyRunning, microphoneInUse }
    private var engine: AVAudioEngine?
    private var observer: (any NSObjectProtocol)?
    private var routeObserver: (any NSObjectProtocol)?
    private var continuation: AsyncStream<Event>.Continuation?

    public init() {}

    public func start(canStart: () -> Bool = { true }) async throws -> AsyncStream<Event> {
        guard engine == nil else { throw Failure.alreadyRunning }
        guard await AVAudioApplication.requestRecordPermission() else { throw Failure.permissionDenied }
        guard canStart() else { throw Failure.microphoneInUse }
        let session = AVAudioSession.sharedInstance()
        // Preserve physical breathing sounds; speech enhancement/voice processing would alter the signal.
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.mixWithOthers, .defaultToSpeaker])
        try session.setPreferredInput(session.availableInputs?.first { $0.portType == .builtInMic })
        try session.setActive(true)
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate >= 8_000, format.sampleRate == format.sampleRate.rounded(),
              format.channelCount > 0 else {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw Failure.invalidInput
        }
        let (stream, continuation) = AsyncStream<Event>.makeStream(bufferingPolicy: .bufferingNewest(4))
        self.continuation = continuation
        let processor = try Processor(sampleRate: Int(format.sampleRate), continuation: continuation)
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { buffer, time in
            processor.process(buffer, time: time)
        }
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            continuation.finish()
            self.continuation = nil
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        self.engine = engine
        observer = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in self?.interrupt() } }
        routeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in self?.interrupt() } }
        return stream
    }

    public func stop(deactivateSession: Bool = true) {
        guard let engine else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        self.engine = nil
        if let observer { NotificationCenter.default.removeObserver(observer) }
        if let routeObserver { NotificationCenter.default.removeObserver(routeObserver) }
        observer = nil
        routeObserver = nil
        continuation?.yield(.stopped)
        continuation?.finish()
        continuation = nil
        if deactivateSession {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    private func interrupt() {
        continuation?.yield(.interrupted)
        stop(deactivateSession: false)
    }
}

/// The audio tap is serial. The engine owns this processor until the tap has been removed.
private final class Processor: @unchecked Sendable {
    private var extractor: AudioFeatureExtractor
    private var expectedSample: AVAudioFramePosition?
    private let sampleRate: Int
    private let continuation: AsyncStream<OvernightAudio.Event>.Continuation

    init(sampleRate: Int, continuation: AsyncStream<OvernightAudio.Event>.Continuation) throws {
        self.sampleRate = sampleRate
        self.extractor = try AudioFeatureExtractor(sampleRate: sampleRate, start: .now)
        self.continuation = continuation
    }

    func process(_ buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0 else { return }
        if let expectedSample, abs(time.sampleTime - expectedSample) > AVAudioFramePosition(sampleRate / 4) {
            continuation.yield(.interrupted)
            // Discard the incomplete epoch; never join audio across a gap.
            extractor = try! AudioFeatureExtractor(sampleRate: sampleRate, start: .now)
        }
        expectedSample = time.sampleTime + AVAudioFramePosition(buffer.frameLength)
        for epoch in extractor.append(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))) {
            continuation.yield(.epoch(epoch))
        }
    }
}
#endif
