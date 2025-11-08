package ma.emsi.bichroujalaleddine;

import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;

import java.util.Scanner;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaifLog {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String llmKey = System.getenv("GEMINIKEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.0-flash-exp")
                .build();

        URL fileUrl = RagNaifLog.class.getResource("/langchain4j.pdf");
        Path path = Paths.get(fileUrl.toURI());
        var parser = new ApacheTikaDocumentParser();
        var document = FileSystemDocumentLoader.loadDocument(path, parser);

        var splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Segments ingérés :");
        for (TextSegment seg : segments) {
            System.out.println("  - " + (seg.text().length() > 60 ? seg.text().substring(0, 60) + "..." : seg.text()));
        }

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("\nAssistant RAG avec logging activé. Tapez 'fin' pour quitter.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) {
                System.out.println("Au revoir !");
                break;
            }
            if (question.trim().isEmpty()) continue;
            String reponse = assistant.chat(question);
            System.out.println("\nRéponse : " + reponse + "\n");
        }
        scanner.close();
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}
