import Foundation
import ImageIO
import Vision

struct OCRLine: Codable {
    let text: String
    let confidence: Float
    let bbox: [Double]
}

struct OCRRecord: Codable {
    let path: String
    let text: String
    let lines: [OCRLine]
    let error: String
}

func usage() -> Never {
    FileHandle.standardError.write(
        "Usage: swift macos_vision_ocr.swift --output OUTPUT.jsonl IMAGE...\n".data(using: .utf8)!
    )
    exit(2)
}

let args = CommandLine.arguments.dropFirst()
var outputPath: String?
var imagePaths: [String] = []
var index = args.startIndex
while index < args.endIndex {
    let arg = args[index]
    if arg == "--output" {
        let next = args.index(after: index)
        if next == args.endIndex {
            usage()
        }
        outputPath = args[next]
        index = args.index(after: next)
    } else {
        imagePaths.append(String(arg))
        index = args.index(after: index)
    }
}

guard let outputPath = outputPath, !imagePaths.isEmpty else {
    usage()
}

func loadCGImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        return nil
    }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}

func recognize(path: String) -> OCRRecord {
    guard let image = loadCGImage(path) else {
        return OCRRecord(path: path, text: "", lines: [], error: "image_load_failed")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return OCRRecord(path: path, text: "", lines: [], error: "vision_request_failed: \(error)")
    }

    let observations = (request.results ?? []).compactMap { observation -> OCRLine? in
        guard let candidate = observation.topCandidates(1).first else {
            return nil
        }
        let box = observation.boundingBox
        return OCRLine(
            text: candidate.string,
            confidence: candidate.confidence,
            bbox: [
                Double(box.origin.x),
                Double(box.origin.y),
                Double(box.size.width),
                Double(box.size.height),
            ]
        )
    }.sorted { left, right in
        let leftTop = left.bbox[1] + left.bbox[3]
        let rightTop = right.bbox[1] + right.bbox[3]
        if abs(leftTop - rightTop) > 0.02 {
            return leftTop > rightTop
        }
        return left.bbox[0] < right.bbox[0]
    }

    return OCRRecord(
        path: path,
        text: observations.map(\.text).joined(separator: "\n"),
        lines: observations,
        error: ""
    )
}

let outputURL = URL(fileURLWithPath: outputPath)
FileManager.default.createFile(atPath: outputURL.path, contents: nil)
guard let output = try? FileHandle(forWritingTo: outputURL) else {
    FileHandle.standardError.write("Cannot open output: \(outputPath)\n".data(using: .utf8)!)
    exit(1)
}
defer {
    try? output.close()
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]

for path in imagePaths {
    let record = recognize(path: path)
    do {
        let data = try encoder.encode(record)
        output.write(data)
        output.write("\n".data(using: .utf8)!)
    } catch {
        FileHandle.standardError.write("JSON encode failed for \(path): \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
