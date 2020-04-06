import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.*;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
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
import java.util.Optional;
import java.util.stream.Collectors;


/**
 * A class that provides functionality for extracting relevant information about imports,
 * packages, etc. from .json files with Java code downloaded from BigQuery
 */
public class Parser {

    public static void main(String[] args) {
        processDataset(args[0], args[1]); // inputFile, outputFile
    }

    /**
     * Parse a single .json file and extract from it the relevant information
     * @param inputFile
     * @param outputFile
     */
    public static void processDataset(String inputFile, String outputFile) {
        JSONParser jsonParser = new JSONParser();
        JavaParser javaParser = new JavaParser();
        List<JSONObject> lines = null;

        try {
            lines = Files.lines(Paths.get(inputFile))
                    .map(file -> parseJSON(jsonParser, file))
                    .map(file -> extractImports(file, javaParser))
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
            for (JSONObject obj: objects)
                file.write(obj.toJSONString() + '\n');
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Initializes a json object to write the data extracted from a single Java file to.
     * @param obj  the object from the original .json file (unparsed)
     * @return     the object that can be filled with parse information
     */
    public static JSONObject constructEmpyTable(JSONObject obj) {
        String[] path = obj.get("path").toString().split("/");
        String name = path[path.length -1].split("\\.")[0];

        JSONObject data = new JSONObject();
        data.put("repo", obj.get("repo_name")); // name of the repository associated with this file
        data.put("name", name); // name of the file itself (also the name of the public class in it)
        data.put("package", ""); // name of the package that this file is in.
        data.put("extends", ""); // the class extended by the public class in the file
        data.put("classImports", new JSONArray()); // all the imported classes\interfaces.
        data.put("packageImports", new JSONArray()); // all the imported packages not including this package.
        data.put("implements", new JSONArray()); // the interfaces implemented by the public class in the file
        return data;
    }


    /**
     * Use Java Parser to extract relevant information from a single Java file
     * @param obj         Java file as a json object (one of many in a .json file)
     * @param javaParser  the Parser object to use
     * @return
     */
    public static JSONObject extractImports(JSONObject obj, JavaParser javaParser) {
        JSONObject data = constructEmpyTable(obj);

        if (obj.get("content") == null) {  // if the file contains no code, do not attempt parsing
            System.out.println("File is empty");
            return data;
        }
        Optional<CompilationUnit> parseResult;
        try {
            parseResult = javaParser.parse((String) obj.get("content")).getResult();
        } catch (Exception e) {
            System.out.println("Bad Java parse error"); // if parsing failed, return an empty object
            return data;
        }
        if (!parseResult.isPresent()) {
            System.out.println("Java parse error");
            return data;
        }

        CompilationUnit ast = parseResult.get();
        DataCollector visitor = new DataCollector();
        visitor.visit(ast, data);
        if (data.get("name").toString().equals("")) {
            // this happens whenever two public non static classes are defined in the same CU
            return constructEmpyTable(obj);
        }
        return data;
    }

    /**
     * A visitor class that fills a json object with information extracted from teh AST
     */
    private static class DataCollector extends VoidVisitorAdapter<JSONObject> {

        @Override
        public void visit(ImportDeclaration id, JSONObject data) {
            super.visit(id, data);
            String name = id.getNameAsString();
            String reducedName = reduceToFile(name);
            if ((id.isAsterisk()) && (reducedName.equals(""))) // package asteriks import
                ((JSONArray) data.get("packageImports")).add(name);
            else
                ((JSONArray) data.get("classImports")).add(reducedName);
        }

        /**
         * Get an identifier like
         * (package.)*Class(.InnerClass)?(.*)?
         * and reduce it to
         * (package.)*Class
         * If this is like (package.)+(*)?, then return null
         * @param identifier
         * @return
         */
        public String reduceToFile(String identifier) {
            String[] parts = identifier.split("\\.|<|>");
            int i = 0;
            int offset = 0;
            while ((i < parts.length) && ((parts[i].length() == 0)||(!Character.isUpperCase(parts[i].charAt(0))))) {
                offset += 1 + parts[i].length();
                i += 1;
            }
            if (i == parts.length)
                return "";
            return identifier.substring(0, offset + parts[i].length());
        }

        /**
         * Record a package declaration
         * @param id   the node in the AST
         * @param data json object to record the package name to
         */
        @Override
        public void visit(PackageDeclaration id, JSONObject data) {
            super.visit(id, data);
            assert("".equals(data.get("package")));
            data.put("package", id.getNameAsString());
        }

        /**
         * Make sure that the name of the main class\interface in the compilation unit matches
         * the name of the file. Also record any information about extended\implemented classes or
         * interfaces
         * @param id   the node in the AST
         * @param data json object to record the information to
         */
        @Override
        public void visit(ClassOrInterfaceDeclaration id, JSONObject data) {
            super.visit(id, data);
            if ((!id.hasModifier(Modifier.Keyword.PUBLIC)) ||
                    (id.hasModifier(Modifier.Keyword.STATIC)) ||
                    (id.getParentNode().get() instanceof ClassOrInterfaceDeclaration))
                return;
            if (!id.getNameAsString().equals(data.get("name"))) { // that can only happen in bad code
                System.out.println(id.getNameAsString() + "\t" + data.get("name"));
                data.put("name", ""); // signal the parser to mark the repository as bad
                return;
            }
            NodeList<ClassOrInterfaceType> implemented = id.getImplementedTypes();
            if (implemented.size() > 0)
                for (ClassOrInterfaceType type: implemented)
                    ((JSONArray) data.get("implements")).add(reduceToFile(type.toString()));
            NodeList<ClassOrInterfaceType> extended = id.getExtendedTypes();
            if (extended.size() > 0) {
                data.put("extends", reduceToFile(extended.get(0).toString()));
            }
        }

        /**
         * Record information about a type reference
         * @param id   the node in the AST
         * @param data json object to record the information to
         */
        @Override
        public void visit(ClassOrInterfaceType id, JSONObject data) {
            super.visit(id, data);
            if (id.getParentNode().get() instanceof ClassOrInterfaceType)
                return;
            ((JSONArray) data.get("classImports")).add(reduceToFile(id.toString()));
        }
    }
}
