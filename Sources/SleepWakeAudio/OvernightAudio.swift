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
    private var starting = false
    private var generation = UUID()
    private var previousInput: AVAudioSessionPortDescription?
    private let spectral: Bool

    public init(spectral: Bool = false) { self.spectral = spectral }

    public func start(canStart: () -> Bool = { true }) async throws -> AsyncStream<Event> {
        guard engine == nil, !starting else { throw Failure.alreadyRunning }
        starting = true
        let attempt = UUID()
        generation = attempt
        defer { starting = false }
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
        let processor = try Processor(sampleRate: Int(format.sampleRate), spectral: spectral, continuation: continuation) { [weak self] in
            Task { @MainActor in
                guard let self, self.generation == attempt else { return }
                self.interrupt()
            }
        }
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { buffer, time in
            processor.process(buffer, time: time)
        }
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            continuation.finish()
            self.continuation = nil
            try? session.setPreferredInput(previousInput)
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
        generation = UUID()
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
        try? AVAudioSession.sharedInstance().setPreferredInput(previousInput)
        previousInput = nil
        if deactivateSession {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    private func interrupt() {
        continuation?.yield(.interrupted)
        stop(deactivateSession: false)
    }
}

/// Bound copied PCM and keep spectrum work off the audio callback. No waveform is written to disk.
private final class Processor: Sendable {
    private struct Packet: Sendable {
        let samples: [Float]
        let sampleTime: AVAudioFramePosition?
        let capturedAt: Date
    }
    private let input: AsyncStream<Packet>.Continuation
    private let worker: Task<Void, Never>
    private let sampleRate: Int

    init(sampleRate: Int, spectral: Bool, continuation: AsyncStream<OvernightAudio.Event>.Continuation,
         interrupted: @escaping @Sendable () -> Void) throws {
        self.sampleRate = sampleRate
        let (packets, input) = AsyncStream<Packet>.makeStream(bufferingPolicy: .bufferingNewest(16))
        self.input = input
        self.worker = Task.detached(priority: .utility) {
            var extractor: AudioFeatureExtractor?
            var timeline = AudioFrameContinuity()
            for await packet in packets {
                guard !Task.isCancelled else { break }
                guard let time = packet.sampleTime, !packet.samples.isEmpty else {
                    interrupted()
                    return
                }
                if !timeline.accept(sampleTime: time, frameCount: packet.samples.count) {
                    interrupted()
                    return
                }
                if extractor == nil {
                    do { extractor = try AudioFeatureExtractor(sampleRate: sampleRate, start: packet.capturedAt, spectral: spectral) }
                    catch { interrupted(); return }
                }
                for epoch in extractor?.append(packet.samples) ?? [] {
                    if case .terminated = continuation.yield(.epoch(epoch)) { return }
                }
            }
            continuation.finish()
        }
    }

    deinit { input.finish(); worker.cancel() }

    func process(_ buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0, time.isSampleTimeValid,
              buffer.format.sampleRate == Double(sampleRate) else {
            input.yield(Packet(samples: [], sampleTime: nil, capturedAt: .now))
            return
        }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        let capturedAt = Date.now.addingTimeInterval(-Double(buffer.frameLength) / buffer.format.sampleRate)
        input.yield(Packet(samples: samples, sampleTime: time.sampleTime, capturedAt: capturedAt))
    }
}
#endif
