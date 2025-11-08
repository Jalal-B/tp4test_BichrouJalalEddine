package ma.emsi.bichroujalaleddine;

import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5_Web {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String llmKey = System.getenv("GEMINI_API_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");

        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ---- PHASE 1 : INGESTION PDF ----
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        URL fileUrl = Test5_Web.class.getResource("/rag.pdf"); // Adapte le chemin si besoin !
        Path path = Paths.get(fileUrl.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(600, 0).split(doc);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        ContentRetriever contentRetrieverLocal = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
        System.out.println("Document PDF ingéré !");

        // ---- PHASE 2 : Moteur Web Tavily ----
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever contentRetrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();
        System.out.println("Moteur web Tavily configuré !");

        // ---- PHASE 3 : QueryRouter ----
        QueryRouter queryRouter = new DefaultQueryRouter(
                contentRetrieverLocal,
                contentRetrieverWeb
        );

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // ---- PHASE 4 : Assistant ----
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("\nAssistant RAG Web prêt (PDF + Web).\nTapez 'quit' pour quitter.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();

            if ("quit".equalsIgnoreCase(question.trim())) {
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
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
        System.out.println("Logging activé !\n");
    }
}
