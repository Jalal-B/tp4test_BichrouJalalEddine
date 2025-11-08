package ma.emsi.bichroujalaleddine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.segment.DefaultDocumentSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiEmbeddingModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.document.parser.apache.tika.ApacheTikaDocumentParser;
import org.apache.tika.Tika;
import dev.langchain4j.data.document.Document;
import java.util.*;

public class Test1_RagNaif {

    public interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {
        // a. Charger le PDF avec le parser Tika officiel LangChain4j
        String pathPdf = "src/main/resources/langchain4j.pdf";
        Document document = FileSystemDocumentLoader.builder()
                .parser(new ApacheTikaDocumentParser())
                .build()
                .loadDocument(pathPdf);

        // b. Découper en segments
        List<TextSegment> segments = DefaultDocumentSplitter.split(document);

        // c. Embedding Gemini
        String apiKey = System.getenv("GEMINIKEY");
        EmbeddingModel embeddingModel = GoogleAiGeminiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("embedding-001")
                .build();

        // d. Embeddings sur tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // e. Stockage en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        for (int i = 0; i < segments.size(); i++) {
            embeddingStore.add(embeddings.get(i), segments.get(i));
        }

        // f. Retriever/Assistant
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .maxResults(4)
                .minScore(0.5)
                .build();

        GoogleAiGeminiChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.25)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .contentRetriever(retriever)
                .build();

        // g. Première question “RAG”
        System.out.println(assistant.chat("Quelle est la signification de 'RAG' ; à quoi ça sert ?"));

        // h. Boucle interactive
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
