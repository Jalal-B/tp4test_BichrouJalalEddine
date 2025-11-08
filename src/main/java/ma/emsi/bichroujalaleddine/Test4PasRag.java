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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4PasRag {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String llmKey = System.getenv("GEMINIKEY");
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // --- Phase 1 : ingestion d'un seul PDF (le support RAG) ---
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        URL fileUrl = Test4PasRag.class.getResource("/langchain4j.pdf"); // modifie le chemin si tu veux TP-4-test.pdf
        Path path = Paths.get(fileUrl.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(600, 0).split(doc);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println(" Phase d'ingestion terminée !\n");

        // --- Classe interne: QueryRouter personnalisé ---
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                PromptTemplate template = PromptTemplate.from(
                        "Est-ce que la requête '{{question}}' concerne l'IA, le RAG, les LLM ou le fine-tuning ? Réponds seulement par 'oui', 'non', ou 'peut-être'."
                );
                Prompt prompt = template.apply(Map.of("question", query.text()));
                String reponse = model.generate(prompt.text());
                System.out.println("🔍 Décision routage ('oui' = RAG, 'non' = pas de RAG): " + reponse.trim());

                if (reponse.trim().toLowerCase().startsWith("non")) {
                    // Pas de RAG
                    return Collections.emptyList();
                } else {
                    // Utiliser le RAG
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant RAG filtré (Pas de RAG hors IA). Tapez 'fin' pour sortir.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) break;
            if (question.trim().isEmpty()) continue;
            String reponse = assistant.chat(question.trim());
            System.out.println("Réponse : " + reponse + "\n");
        }
        scanner.close();

        // Pour un script de test automatique comme dans l'énoncé :
        /*
        System.out.println("Test 1 : 'Bonjour' (pas de RAG attendu)");
        System.out.println("→ " + assistant.chat("Bonjour") + "\n");
        System.out.println("Test 2 : 'Qu'est-ce que le RAG ?' (RAG attendu)");
        System.out.println("→ " + assistant.chat("Qu'est-ce que le RAG ?") + "\n");
        System.out.println("Test 3 : 'Quelle est la capitale de la France ?' (pas de RAG attendu)");
        System.out.println("→ " + assistant.chat("Quelle est la capitale de la France ?"));
        */
    }

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }
}
