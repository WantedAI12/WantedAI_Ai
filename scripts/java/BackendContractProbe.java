import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import com.perfumeryaicore.global.client.dto.FormulaGenerationResponse;
import com.perfumeryaicore.global.client.dto.LotionDesignResponse;

/** Read-only probe compiled against unmodified backend DTO source files. */
public class BackendContractProbe {
    public static void main(String[] args) throws Exception {
        JsonMapper mapper = JsonMapper.builder().build();
        var results = new ArrayList<Map<String, Object>>();
        for (int i = 1; i < args.length; i += 2) {
            String type = args[i];
            Path path = Path.of(args[i + 1]);
            var row = new LinkedHashMap<String, Object>();
            row.put("class", type);
            row.put("file", path.toString());
            row.put("bytes", Files.size(path));
            row.put("within_backend_8MiB", Files.size(path) <= 8 * 1024 * 1024);
            String raw = Files.readString(path);
            try {
                Object value = mapper.readValue(raw,
                    Class.forName("com.perfumeryaicore.global.client.dto." + type));
                row.put("parsed", true);
                if (value instanceof FormulaGenerationResponse formula) {
                    row.put("candidate_stored_by_backend", !formula.isNoSafeMatch() && formula.recipeSize() > 0);
                    row.put("recipe_size", formula.recipeSize());
                    row.put("status", formula.status());
                    JsonNode confidence = mapper.readTree(raw).get("simulation_confidence");
                    try {
                        row.put("simulation_confidence_projected", confidence == null || confidence.isNull() ? null : confidence.asDouble());
                        row.put("prediction_mapper_numeric_projection_ok", true);
                    } catch (RuntimeException e) {
                        row.put("prediction_mapper_numeric_projection_ok", false);
                        row.put("numeric_projection_error", e.getClass().getSimpleName());
                    }
                } else if (value instanceof LotionDesignResponse lotion) {
                    row.put("candidate_stored_by_backend", lotion.isUsableCandidate());
                    row.put("recipe_size", lotion.recipeSize());
                    row.put("profile_target_met", lotion.profileTargetMet());
                    row.put("status", lotion.status());
                }
            } catch (RuntimeException e) {
                row.put("parsed", false);
                row.put("error", e.getClass().getSimpleName() + ": " + e.getMessage());
            }
            results.add(row);
        }
        var report = new LinkedHashMap<String, Object>();
        report.put("backend_source_modified", false);
        report.put("scope", "actual_Jackson_DTO_and_selection_methods_not_full_Spring_application");
        report.put("java_version", System.getProperty("java.version"));
        report.put("cases", results);
        Path target = Path.of(args[0]);
        Files.createDirectories(target.getParent());
        Files.writeString(target, mapper.writerWithDefaultPrettyPrinter().writeValueAsString(report));
        System.out.println(mapper.writeValueAsString(report));
    }
}
