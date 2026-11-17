import Foundation

/// ZIP32 stored entries with streaming CRC/data descriptors. Large recordings never load into memory at once.
enum SleepNightArchive {
    enum Failure: Error { case invalidFile, tooLarge }
    private struct Entry { var name: Data; var crc: UInt32; var size: UInt32; var offset: UInt32 }
    private static let crcTable: [UInt32] = (0..<256).map { number in
        var c = UInt32(number)
        for _ in 0..<8 { c = (c >> 1) ^ ((c & 1) == 1 ? 0xedb88320 : 0) }
        return c
    }

    static func write(_ files: [(String, URL)], to destination: URL) throws {
        guard files.count < 65_536, Set(files.map(\.0)).count == files.count else { throw Failure.tooLarge }
        guard !FileManager.default.fileExists(atPath: destination.path),
              FileManager.default.createFile(atPath: destination.path, contents: nil) else { throw Failure.invalidFile }
        let output = try FileHandle(forWritingTo: destination)
        var success = false
        defer { try? output.close(); if !success { try? FileManager.default.removeItem(at: destination) } }
        var position: UInt64 = 0
        func write(_ data: Data) throws {
            guard position + UInt64(data.count) < UInt64(UInt32.max) else { throw Failure.tooLarge }
            try output.write(contentsOf: data); position += UInt64(data.count)
        }
        var entries: [Entry] = []
        for (name, url) in files {
            let resource = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard resource.isRegularFile == true, resource.isSymbolicLink != true,
                  !name.hasPrefix("/"), !name.split(separator: "/").contains("..") else { throw Failure.invalidFile }
            let bytes = Data(name.utf8)
            guard bytes.count <= Int(UInt16.max) else { throw Failure.tooLarge }
            let offset = UInt32(position)
            var header = Data()
            header.le(UInt32(0x04034b50)); header.le(UInt16(20)); header.le(UInt16(0x0808)); header.le(UInt16(0))
            header.le(UInt16(0)); header.le(UInt16(33)); header.le(UInt32(0)); header.le(UInt32(0)); header.le(UInt32(0))
            header.le(UInt16(bytes.count)); header.le(UInt16(0)); header.append(bytes)
            try write(header)
            let input = try FileHandle(forReadingFrom: url)
            var crc = UInt32.max; var size: UInt64 = 0
            do {
                while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
                    size += UInt64(chunk.count)
                    guard size < UInt64(UInt32.max) else { throw Failure.tooLarge }
                    for byte in chunk { crc = crcTable[Int((crc ^ UInt32(byte)) & 255)] ^ (crc >> 8) }
                    try write(chunk)
                }
                try input.close()
            } catch { try? input.close(); throw error }
            crc ^= UInt32.max
            var descriptor = Data(); descriptor.le(UInt32(0x08074b50)); descriptor.le(crc)
            descriptor.le(UInt32(size)); descriptor.le(UInt32(size)); try write(descriptor)
            entries.append(Entry(name: bytes, crc: crc, size: UInt32(size), offset: offset))
        }
        let centralStart = UInt32(position)
        for entry in entries {
            var h = Data(); h.le(UInt32(0x02014b50)); h.le(UInt16(20)); h.le(UInt16(20)); h.le(UInt16(0x0808))
            h.le(UInt16(0)); h.le(UInt16(0)); h.le(UInt16(33)); h.le(entry.crc); h.le(entry.size); h.le(entry.size)
            h.le(UInt16(entry.name.count)); h.le(UInt16(0)); h.le(UInt16(0)); h.le(UInt16(0)); h.le(UInt16(0))
            h.le(UInt32(0)); h.le(entry.offset); h.append(entry.name); try write(h)
        }
        let centralSize = UInt32(position) - centralStart
        var end = Data(); end.le(UInt32(0x06054b50)); end.le(UInt16(0)); end.le(UInt16(0))
        end.le(UInt16(entries.count)); end.le(UInt16(entries.count)); end.le(centralSize); end.le(centralStart); end.le(UInt16(0))
        try write(end); try output.synchronize()
        #if os(iOS)
        try FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], ofItemAtPath: destination.path)
        #endif
        success = true
    }
}

private extension Data {
    mutating func le<T: FixedWidthInteger>(_ number: T) {
        var value = number.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }
}
