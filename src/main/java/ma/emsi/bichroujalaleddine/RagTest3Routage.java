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
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagTest3Routage {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String apiKey = System.getenv("GEMINIKEY"); // ou "GEMINIKEY" selon ton env
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true) // BONUS : le log Gemini apparaît dans la console !
                .build();

        // --- EmbeddingStores/Retrievers pour les 2 PDFs ---
        ContentRetriever retrieverIA = buildRetriever("/langchain4j.pdf");
        ContentRetriever retrieverGestion = buildRetriever("/LAB 1 Gestion Commerciale.pdf");

        // Descriptions utilisées pour le rapport bonus
        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();
        descriptions.put(retrieverIA, "Support sur l’IA, RAG, LangChain4j, modèles de langage.");
        descriptions.put(retrieverGestion, "Cours de gestion commerciale, ERP, pratiques comptables.");

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                .chatMemory(chatMemory)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("\nAssistant Routage TP3. Tapez 'fin' pour quitter et copier le log JSON routage Gemini pour le bonus.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) break;
            if (question.trim().isEmpty()) continue;

            System.out.println("Voulez-vous :");
            System.out.println("1 - Chercher dans le PDF IA (langchain4j.pdf)");
            System.out.println("2 - Chercher dans le PDF Gestion (LAB 1 Gestion Commerciale.pdf)");
            String choix = scanner.nextLine();

            ContentRetriever retriever = null;
            if ("1".equals(choix.trim())) {
                retriever = retrieverIA;
            } else if ("2".equals(choix.trim())) {
                retriever = retrieverGestion;
            } else {
                System.out.println("Choix invalide !");
                continue;
            }

            // DEBUG/bonus : Affiche les segments pertinents du ContentRetriever
            List<Content> results = retriever.retrieve(Query.from(question));
            System.out.println("--- Segments pertinents trouvés ---");
            for (Content content : results) {
                String segText = content.toString();
                segText = segText.length() > 120 ? segText.substring(0,120) + "..." : segText;
                System.out.println(segText);
            }
            if (results.isEmpty()) System.out.println("Aucun segment ne match !");

            // Appel Gemini/LLM : la partie qui déclenche le log JSON "Sending request to Gemini"
            String reponse = assistant.chat(question);
            System.out.println("\nRéponse Gemini : " + reponse + "\n");
        }
        scanner.close();
    }

    // INGESTION D'UN PDF → ContentRetriever
    private static ContentRetriever buildRetriever(String resourceName) throws Exception {
        URL fileUrl = RagTest3Routage.class.getResource(resourceName);
        if (fileUrl == null) throw new RuntimeException("Fichier introuvable : " + resourceName);
        Path path = Paths.get(fileUrl.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(500, 0).split(doc);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        System.out.println("Document '" + resourceName + "' ingéré. Nb segments : " + segments.size());
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.1)
                .build();
    }

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }
}
