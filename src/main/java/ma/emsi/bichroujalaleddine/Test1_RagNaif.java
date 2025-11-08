package ma.emsi.bichroujalaleddine;

import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.*;
import dev.langchain4j.data.segment.*;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.bedrock.BedrockTitanEmbeddingModel;
import dev.langchain4j.store.embedding.inmemory.*;
import dev.langchain4j.store.embedding.*;
import dev.langchain4j.rag.content.retriever.*;
import dev.langchain4j.service.*;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import java.io.File;
import java.util.*;

public class Test1_RagNaif {

    public interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {
        // a. Charge le document PDF
        String nomDocument = "chemin/vers/toncours.pdf"; // mets le nom de ton vrai fichier
        Document document = FileSystemDocumentLoader.loadDocument(nomDocument);

        // b. Découpe le document
        List<TextSegment> segments = DefaultDocumentSplitter.split(document);

        // c. Créer le modèle d'embeddings Gemini
        String embeddingApiKey = System.getenv("GEMINIKEY");
        EmbeddingModel embeddingModel = GoogleAiGeminiChatModel.embeddingModelBuilder().apiKey(embeddingApiKey).build();

        // d. Genère tous les embeddings
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // e. Stock/ngeste toutes les embeddings dans le store
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        for (int i = 0; i < segments.size(); i++) {
            embeddingStore.add(embeddings.get(i), segments.get(i));
        }

        // f. Crée le ContentRetriever avec ce store et un assistant
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .maxResults(4) // adapte selon ce que tu veux
                .minScore(0.5)
                .build();

        String llmKey = System.getenv("GEMINIKEY");
        GoogleAiGeminiChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.25)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .contentRetriever(retriever)
                .build();

        // g. Pose la première question
        System.out.println(assistant.chat("Quelle est la signification de 'RAG' ; à quoi ça sert ?"));

        // h. Conversation interactive
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nQuestion ('fin' pour arrêter) : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("fin"))
                break;
            System.out.println(assistant.chat(question));
        }
    }
}
