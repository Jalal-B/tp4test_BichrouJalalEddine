package ma.emsi.bichroujalaleddine;


import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
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
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.Scanner;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;


public class RagNaif {

    public static void main(String[] args) throws Exception {
        String GEMINI_KEY = System.getenv("GEMINIKEY");
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_KEY)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.23)
                .build();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        Document docCours = chargerDocument("langchain4j.pdf");
        List<TextSegment> decoupes = decouperDocument(docCours, 512);
        List<Embedding> embList = genererEmbeddings(embeddingModel, decoupes);

        EmbeddingStore<TextSegment> memStore = new InMemoryEmbeddingStore<>();
        memStore.addAll(embList, decoupes);

        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(memStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory mem = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(mem)
                .contentRetriever(retriever)
                .build();

        System.out.println("Assistant prêt ! Pour quitter, tapez 'stop'.");

        Scanner input = new Scanner(System.in);
        while (true) {
            System.out.print("\nVotre question : ");
            String q = input.nextLine();
            if ("stop".equalsIgnoreCase(q.trim())) {
                System.out.println("Session terminée !");
                break;
            }
            if (q.trim().isEmpty()) continue;
            String result = assistant.chat(q);
            System.out.println("Réponse :\n" + result);
        }
        input.close();
    }

    private static Document chargerDocument(String fileName) throws Exception {
        URL fileURL = RagNaif.class.getResource("/" + fileName);
        Path path = Paths.get(fileURL.toURI());
        DocumentParser parser = new ApacheTikaDocumentParser();
        return FileSystemDocumentLoader.loadDocument(path, parser);
    }

    private static List<TextSegment> decouperDocument(Document doc, int taille) {
        DocumentSplitter splitter = DocumentSplitters.recursive(taille, 0);
        return splitter.split(doc);
    }

    private static List<Embedding> genererEmbeddings(EmbeddingModel model, List<TextSegment> segments) {
        Response<List<Embedding>> embResponse = model.embedAll(segments);
        return embResponse.content();
    }
}
