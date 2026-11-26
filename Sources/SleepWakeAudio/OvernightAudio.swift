#if os(iOS)
import AVFAudio
import Foundation
import SleepWakeKit

/// Explicitly started observation. Optional local recording is confined to the worker; no network access.
@MainActor
public final class OvernightAudio {
    public enum Event: Sendable {
        case epoch(AudioEpoch)
        case interrupted
        case stopped
        case failed
    }
    public enum Failure: Error { case permissionDenied, invalidInput, alreadyRunning, microphoneInUse }
    private var engine: AVAudioEngine?
    private var observer: (any NSObjectProtocol)?
    private var routeObserver: (any NSObjectProtocol)?
    private var continuation: AsyncStream<Event>.Continuation?
    private var starting = false
    private var generation = UUID()
    private var previousInput: AVAudioSessionPortDescription?
    private var processor: Processor?
    private var finishing: Task<Void, Never>?
    public private(set) var inputSampleRate: Int?
    public static let recordingSampleRate = 16_000
    private let spectral: Bool
    private let acousticFrames: Bool

    public init(spectral: Bool = false, acousticFrames: Bool = false) {
        self.spectral = spectral
        self.acousticFrames = acousticFrames
    }

    public func start(audioURL: URL? = nil, canStart: () -> Bool = { true }) async throws -> AsyncStream<Event> {
        guard engine == nil, !starting else { throw Failure.alreadyRunning }
        starting = true
        let attempt = UUID()
        generation = attempt
        defer { starting = false }
        if let finishing { await finishing.value; self.finishing = nil }
        guard await AVAudioApplication.requestRecordPermission() else { throw Failure.permissionDenied }
        try Task.checkCancellation()
        guard generation == attempt else { throw CancellationError() }
        guard canStart() else { throw Failure.microphoneInUse }
        let session = AVAudioSession.sharedInstance()
        previousInput = session.preferredInput
        // Preserve physical breathing sounds; speech enhancement/voice processing would alter the signal.
        do {
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.mixWithOthers, .defaultToSpeaker])
            try session.setPreferredInput(session.availableInputs?.first { $0.portType == .builtInMic })
            try session.setActive(true)
        } catch {
            try? session.setPreferredInput(previousInput)
            previousInput = nil
            throw error
        }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard (8_000...192_000).contains(format.sampleRate), format.sampleRate == format.sampleRate.rounded(),
              format.channelCount > 0, format.commonFormat == .pcmFormatFloat32, !format.isInterleaved else {
            try? session.setPreferredInput(previousInput)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw Failure.invalidInput
        }
        let (stream, continuation) = AsyncStream<Event>.makeStream(bufferingPolicy: .bufferingNewest(4))
        self.continuation = continuation
        continuation.onTermination = { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == attempt else { return }
                self.stop()
            }
        }
        let processor: Processor
        do {
            processor = try Processor(sampleRate: Int(format.sampleRate), audioURL: audioURL, spectral: spectral, acousticFrames: acousticFrames, continuation: continuation) { [weak self] failed in
                Task { @MainActor in
                    guard let self, self.generation == attempt else { return }
                    self.stop()
                }
            }
        } catch {
            continuation.finish(); self.continuation = nil
            try? session.setPreferredInput(previousInput)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        self.processor = processor
        input.installTap(onBus: 0, bufferSize: 4_096, format: format, block: Self.tap(processor))
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            processor.finish(); self.processor = nil
            continuation.finish()
            self.continuation = nil
            try? session.setPreferredInput(previousInput)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        self.engine = engine
        inputSampleRate = Int(format.sampleRate)
        let expectedRate = format.sampleRate
        observer = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in self?.interrupt() } }
        routeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in Task { @MainActor in
            let session = AVAudioSession.sharedInstance()
            guard session.mode != .measurement || session.category != .playAndRecord
                    || session.currentRoute.inputs.first?.portType != .builtInMic
                    || session.sampleRate != expectedRate else { return }
            self?.interrupt()
        } }
        return stream
    }

    public func stop(deactivateSession: Bool = true) {
        generation = UUID()
        guard let engine else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        self.engine = nil
        processor?.finish()
        finishing = processor?.worker
        processor = nil
        if let observer { NotificationCenter.default.removeObserver(observer) }
        if let routeObserver { NotificationCenter.default.removeObserver(routeObserver) }
        observer = nil
        routeObserver = nil
        continuation = nil
        try? AVAudioSession.sharedInstance().setPreferredInput(previousInput)
        previousInput = nil
        if deactivateSession {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    /// Stops new input immediately, then waits for queued PCM, file headers and final epochs to finish.
    public func finish(deactivateSession: Bool = true) async {
        stop(deactivateSession: deactivateSession)
        await finishing?.value
        finishing = nil
    }

    /// Built outside the main actor: the audio thread calls it, and a main-actor closure traps there.
    nonisolated private static func tap(_ processor: Processor) -> AVAudioNodeTapBlock {
        { buffer, time in processor.process(buffer, time: time) }
    }

    private func interrupt() {
        continuation?.yield(.interrupted)
        stop(deactivateSession: false)
    }
}

/// Bound copied PCM and keep file conversion/spectrum work off the audio callback.
private final class Processor: Sendable {
    private struct Packet: Sendable {
        let samples: [Float]
        let sampleTime: AVAudioFramePosition?
        let capturedAt: Date
    }
    private let input: AsyncStream<Packet>.Continuation
    let worker: Task<Void, Never>
    private let sampleRate: Int

    init(sampleRate: Int, audioURL: URL?, spectral: Bool, acousticFrames: Bool, continuation: AsyncStream<OvernightAudio.Event>.Continuation,
         interrupted: @escaping @Sendable (Bool) -> Void) throws {
        self.sampleRate = sampleRate
        let (packets, input) = AsyncStream<Packet>.makeStream(bufferingPolicy: .bufferingNewest(16))
        self.input = input
        let writer = try audioURL.map { try NightAudioWriter(url: $0, sampleRate: sampleRate) }
        self.worker = Task.detached(priority: .utility) {
            func signal(_ failed: Bool) {
                continuation.yield(failed ? .failed : .interrupted)
                interrupted(failed)
            }
            defer {
                do { try writer?.finish() } catch { signal(true) }
                continuation.yield(.stopped)
                continuation.finish()
            }
            var extractor: AudioFeatureExtractor?
            var timeline = AudioFrameContinuity()
            for await packet in packets {
                guard !Task.isCancelled else { break }
                guard let time = packet.sampleTime, !packet.samples.isEmpty else {
                    signal(false)
                    return
                }
                if !timeline.accept(sampleTime: time, frameCount: packet.samples.count) {
                    signal(false)
                    return
                }
                if extractor == nil {
                    do { extractor = try AudioFeatureExtractor(sampleRate: sampleRate, start: packet.capturedAt, spectral: spectral, acousticFrames: acousticFrames) }
                    catch { signal(false); return }
                }
                do { try writer?.append(packet.samples) }
                catch { signal(true); return }
                for epoch in extractor?.append(packet.samples) ?? [] {
                    switch continuation.yield(.epoch(epoch)) {
                    case .terminated: return
                    case .dropped: signal(false); return
                    case .enqueued: break
                    @unknown default: signal(false); return
                    }
                }
            }
        }
    }

    deinit { input.finish() }
    func finish() { input.finish() }

    func process(_ buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0, time.isSampleTimeValid,
              buffer.format.sampleRate == Double(sampleRate) else {
            input.yield(Packet(samples: [], sampleTime: nil, capturedAt: .now))
            return
        }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        let capturedAt = time.isHostTimeValid
            ? Date.now.addingTimeInterval(AVAudioTime.seconds(forHostTime: time.hostTime) - ProcessInfo.processInfo.systemUptime)
            : Date.now.addingTimeInterval(-Double(buffer.frameLength) / buffer.format.sampleRate)
        input.yield(Packet(samples: samples, sampleTime: time.sampleTime, capturedAt: capturedAt))
    }
}
#endif
