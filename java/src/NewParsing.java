import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class NewParsing {

    public static final String PREFIX = "/Volumes/My Passport/BigQuery/contents.final-0000000000";
    public static final String[] FILES = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26"};

    public static void main(String[] args) {
        for (String file: FILES)
            processDataset(PREFIX + file + ".json", PREFIX + file + "Out.json");
    }

    public static void processDataset(String inputFile, String outputFile) {
        JSONParser jsonParser = new JSONParser();
        List<JSONObject> lines = null;

        Rectable x = new Rectangle

        try {
            lines = Files.lines(Paths.get(inputFile))
                    .map(file -> parseJSON(jsonParser, file))
                    .filter(Objects::nonNull)
                    .map(file -> extractImports(file))
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList()); // get the list of files
        } catch (IOException e) {
            e.printStackTrace();
        }

        saveJSON(lines, outputFile);
    }


    /***
     * A wrapper around jsonParser.parse with exception caught (so that it can be used in a stream)
     * @param jsonParser A JSONParser
     * @param line       A Json object (a line in the file which is read)
     * @return
     */
    public static JSONObject parseJSON(JSONParser jsonParser, String line) {
        try {
            return (JSONObject) jsonParser.parse(line);
        } catch (ParseException e) {
            return null;
        }
    }


    /**
     * Saves the json data back to a file
     * @param objects
     * @param outputFile
     */
    public static void saveJSON(List<JSONObject> objects, String outputFile) {
        try (FileWriter file = new FileWriter(outputFile)) {
            if (objects != null)
                for (JSONObject obj: objects)
                    file.write(obj.toJSONString() + '\n');
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static JSONObject extractImports(JSONObject obj) {
        if (obj.get("content") == null)
                return null;
        CompilationUnit ast;
        try {
            ast = JavaParser.parse((String) obj.get("content"));
        } catch (Exception | AssertionError e) {
            System.out.println("Parsing failed");
            return null;
        }

        JSONObject data = new JSONObject();
        data.put("repo", obj.get("repo_name"));
        data.put("imports", new JSONArray());
        data.put("classes", new JSONArray());
        DataCollector visitor = new DataCollector();
        visitor.visit(ast, data);
        return data;
    }

    private static class DataCollector extends VoidVisitorAdapter<JSONObject> {

        @Override
        public void visit(ImportDeclaration id, JSONObject data) {
            super.visit(id, data);
            ((JSONArray) data.get("imports")).add(id.getNameAsString());
        }

        @Override
        public void visit(PackageDeclaration id, JSONObject data) {
            super.visit(id, data);
            data.put("package", id.getNameAsString());
        }

        @Override
        public void visit(ClassOrInterfaceDeclaration id, JSONObject data) {
            super.visit(id, data);
            ((JSONArray) data.get("classes")).add(id.getNameAsString());
        }
    }
}
