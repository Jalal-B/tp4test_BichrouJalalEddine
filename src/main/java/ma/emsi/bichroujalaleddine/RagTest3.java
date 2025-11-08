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
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagTest3 {

    public static void main(String[] args) throws Exception {
        String apiKey = System.getenv("GEMINIKEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 📚 Charger deux sources PDF différentes
        URL pdfUrl1 = RagTest3.class.getResource("/langchain4j.pdf");
        Path path1 = Paths.get(pdfUrl1.toURI());
        URL pdfUrl2 = RagTest3.class.getResource("/LAB 1 Gestion Commerciale.pdf");
        Path path2 = Paths.get(pdfUrl2.toURI());

        Document doc1 = FileSystemDocumentLoader.loadDocument(path1, new ApacheTikaDocumentParser());
        Document doc2 = FileSystemDocumentLoader.loadDocument(path2, new ApacheTikaDocumentParser());

        List<TextSegment> chunks1 = DocumentSplitters.recursive(600, 0).split(doc1);
        List<TextSegment> chunks2 = DocumentSplitters.recursive(600, 0).split(doc2);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<dev.langchain4j.data.embedding.Embedding> embeddings1 = embeddingModel.embedAll(chunks1).content();
        List<dev.langchain4j.data.embedding.Embedding> embeddings2 = embeddingModel.embedAll(chunks2).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings1, chunks1);
        embeddingStore.addAll(embeddings2, chunks2);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(retriever)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant Test3 ok. Tapez 'fin' pour quitter.");
        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) break;
            if (question.trim().isEmpty()) continue;
            String reponse = assistant.chat(question);
            System.out.println("Réponse : " + reponse + "\n");
        }
        scanner.close();
    }
}
