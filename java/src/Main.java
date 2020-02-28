import com.github.javaparser.*;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.type.ArrayType;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.type.WildcardType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Main {

    public final static String DATA_FILE = "/Users/alexanderfedchin/Documents/ngrams/content_example.json";
    public final static ArrayList<String> DELIM = new ArrayList(Arrays.asList(new String[] {" ", "\t", "\n", "\r", "\r\n"}));
    public final static ArrayList<Integer> COMMENTS = new ArrayList(Arrays.asList(new Integer[] {9, 8, 5}));

    public static void main(String[] args) throws Exception {
        JSONParser jsonParser = new JSONParser();
        List<String> repositories = Files.lines(Paths.get(DATA_FILE))
                .map(file -> parseJSON(jsonParser, file))
                .map(file -> ((String) file.get("repo_path")).split("/")[0])
                .distinct()
                .collect(Collectors.toList()); // get the list of files

        Collections.shuffle(repositories);
        List<String> trainRepo = repositories.subList(0, (int)(0.8 * repositories.size()));
        List<String> testRepo = repositories.subList((int)(0.8 * repositories.size()), repositories.size());

        List<String> trainFiles = Files.lines(Paths.get(DATA_FILE))
                .map(file -> parseJSON(jsonParser, file))
                .filter(file -> trainRepo.contains(( (String) file.get("repo_path")).split("/")[0]))
                .map(file -> (String) file.get("content"))
                .filter(Objects::nonNull)
                .collect(Collectors.toList()); // get the list of files

        List<String> testFiles = Files.lines(Paths.get(DATA_FILE))
                .map(file -> parseJSON(jsonParser, file))
                .filter(file -> testRepo.contains(( (String) file.get("repo_path")).split("/")[0]))
                .map(file -> (String) file.get("content"))
                .filter(Objects::nonNull)
                .collect(Collectors.toList()); // get the list of files

        System.out.println(trainFiles.size() + " " + testFiles.size());

        processData(trainFiles, 0, trainFiles.size(),
                "/Users/alexanderfedchin/Documents/ngrams/train.tsv",
                "/Users/alexanderfedchin/Documents/ngrams/trainType.tsv");
        processData(testFiles, 0, testFiles.size(),
                "/Users/alexanderfedchin/Documents/ngrams/test.tsv",
                "/Users/alexanderfedchin/Documents/ngrams/testType.tsv");
    }


    static void saveAST(CompilationUnit ast, PrintWriter pw, boolean writeTokenType) {
        StringBuilder line = new StringBuilder();
        JavaToken token = ast.getTokenRange().get().getBegin();
        do {
            String text = token.getText();
            if ((!DELIM.contains(text))&&(!COMMENTS.contains(token.getKind()))) {
                if (writeTokenType)
                    line.append(token.getKind() + " ");
                else
                    line.append(text + " ");
            }
            token = token.getNextToken().orElse(null);
        } while (token != null);
        pw.println(line);
    }

    static CompilationUnit getAST(String file) {
        CompilationUnit ast;
        Replacer visitor = new Replacer();
        try {
            ast = JavaParser.parse(file);
            removeComments(ast); // get rid of comments
            ast.removeComment();
        } catch (Exception ex) {
            //System.out.println("Parse issue");
            return null;
        }
        visitor.visit(ast, null);
        return ast;
    }

    static void processData(List<String> files, int begin, int end, String pathToTokenFile,
                            String pathToTypeFile) {
        int count = 0;
        int parsed = 0;
        try {
            PrintWriter pwtoken = new PrintWriter(Files.newBufferedWriter(Paths.get(pathToTokenFile)));
            PrintWriter pwtype = new PrintWriter(Files.newBufferedWriter(Paths.get(pathToTypeFile)));
            for (int i = begin; i < end; i++) {
                if ((double)(i-begin-count)/(end-begin) >= 0.05) {
                    System.out.println(count + " (" + (double)(i-begin)/(end-begin) + ") done...");
                    count = i - begin;
                }
                CompilationUnit ast = getAST(files.get(i));
                if (ast != null) {
                    saveAST(ast, pwtoken, false);
                    saveAST(ast, pwtype, true);
                    parsed++;
                }
            }
            pwtoken.close();
            pwtype.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(parsed + " (" + (double)(parsed)/(end-begin) + ") parsed.");
    }

    static void removeComments(Node node) {
        for (Node child : node.getChildNodes()) {
            child.removeComment();
            removeComments(child);
        }
    }

    public static void getStats(List<String> files) {
        ImportNamePrinter inp = new ImportNamePrinter();
        HashMap<String, HashMap<String, Integer>> coocc = new HashMap<>();
        HashMap<String, Integer> totals = new HashMap<>();
        long total = 0;
        List<String> imports;


        int count = 0;
        for (String file: files) {
            count++;
            if (count % 5000 == 0) {
                System.out.println(count + " " + files.size());
            }
            imports = new ArrayList<>();
            try {
                inp.visit(JavaParser.parse(file), imports);
            } catch (Exception ex) {}
            imports = imports.stream().distinct().collect(Collectors.toList());
            total += imports.size();
            for (int i = 0; i < imports.size(); i++)
                for (int j = i + 1; j < imports.size(); j++) {
                    String key0 = imports.get(i);
                    String key1 = imports.get(j);
                    totals.put(key0, totals.getOrDefault(key0, 0) + 1);
                    totals.put(key1, totals.getOrDefault(key1, 0) + 1);

                    HashMap<String, Integer> map = coocc.getOrDefault(key0, new HashMap<>());
                    map.put(key1, map.getOrDefault(key1, 0)+ 1);
                    coocc.putIfAbsent(key0, map);

                    map = coocc.getOrDefault(key1, new HashMap<>());
                    map.put(key0, map.getOrDefault(key0, 0)+ 1);
                    coocc.putIfAbsent(key1, map);
                }
        }

        List<TestStats> pValues = getChiSquared(total, totals, coocc);
        Collections.sort(pValues, Comparator.comparingDouble(TestStats::getpValue));
        for(int i = pValues.size() - 1; i > pValues.size() - 100; i--){
            System.out.println(pValues.get(i));
        }

        System.out.println();
        int counter = 0;
        int i = pValues.size() - 1;
        while ((i >= 0) && (counter < 100)) {
            if ((pValues.get(i).getCooc() > 100)&&(pValues.get(i).only0 < 10000)) {
                System.out.println(pValues.get(i));
                counter++;
            }
            i--;
        }

        Collections.sort(pValues, Comparator.comparingLong(TestStats::getCooc));
        System.out.println();
        for(i = pValues.size() - 1; i > pValues.size() - 100; i--){
            System.out.println(pValues.get(i));
        }
        System.out.println(5);
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
     * Iterate over all the pairs of classes that occured together at least ones, calculate
     * chi-squared test of independence for each pair and return the results as a list of
     * TestStats instances
     * @param total   total number of import statement in the corpus
     * @param totals  totals.get("java.utils.List") is the number of imports the corresponding class
     * @param coocc   coocc.get("class0").get("class1") is the number of classes in which both
     *                class0 and class1 are imported
     * @return
     */
    public static List<TestStats> getChiSquared(long total,
                                                HashMap<String, Integer> totals,
                                                HashMap<String, HashMap<String, Integer>> coocc) {
        List<TestStats> pValues = new ArrayList<>();

        for (String key0: coocc.keySet()) // iterate over all pairs of import statements
            for (String key1: coocc.get(key0).keySet()) {

                // this is to make sure that the test is performed only once for each pair:
                if (totals.get(key0) < totals.get(key1))
                    continue;
                else if ((totals.get(key0) == totals.get(key1)) && (key0.compareTo(key1) < 0))
                    continue;

                pValues.add(new TestStats(key0 + ' ' + key1,
                            coocc.get(key0).get(key1),
                            totals.get(key0),
                            totals.get(key1),
                            total));
            }

        return pValues;
    }


    /**
     * A class to perform and store statistics about a chi-square independent test
     */
    private static class TestStats {
        static ChiSquareTest chi = new ChiSquareTest(); // for performing the test
        double pValue; // the p-value as returned by chi-square test
        String name; // names of both classes compared concatenated
        long cooc, only0, only1; // number of cooccurances,
        // number of class0 instances without class1, and the reverse

        TestStats(String name, long cooc, long total0, long total1, long total) {
            this.name = name;

            this.only0 = total0 - cooc;
            this.only1 = total1 - cooc;
            this.cooc = cooc;
            long none = total - cooc - only0 - only1;

            this.pValue = chi.chiSquare(new long[][] {{cooc, only0}, {only1, none}});
        }

        double getpValue() {
            return pValue;
        }

        long getCooc() {
            return cooc;
        }

        @Override
        public String toString() {
            return name + " " + cooc + " " + only0 + " " + only1 + " " + pValue;
        }
    }

    public static class ImportNamePrinter extends AbstractVisitor<String> {

        @Override
        public String visitTypeName(SimpleName id) {
            return id.asString();
        }
    }

    /***
     * A Visitor that locates all the ImportDeclarations in a file and adds them all to a list
     */
    private abstract static class AbstractVisitor<T> extends VoidVisitorAdapter<List<T>> {
        static Pattern p = Pattern.compile("[A-Z][A-Za-z0-9]*");

        @Override
        public void visit(ClassOrInterfaceType id, List<T> names) {
            super.visit(id, names);  // not necessary, but documentation suggests using it anyway
            recursiveVisit(id, names);
        }

        public void recursiveVisit(Node id, List<T> names) {
            for (Node node: id.getChildNodes())
                if ((node instanceof ClassOrInterfaceType) || (node instanceof ArrayType)) {
                    recursiveVisit(node, names);
                } else if (node instanceof SimpleName) {
                    SimpleName simpleName = (SimpleName)node;
                    if (!nameCanBeVisited(simpleName))
                        continue;
                    names.add(visitTypeName(simpleName));
                } else if ((node instanceof WildcardType) ||
                        (node instanceof PrimitiveType) ||
                        (node instanceof MarkerAnnotationExpr) ||
                        (node instanceof SingleMemberAnnotationExpr)) {

                } else {
                    System.out.println("Problem here");
                }
        }

        public boolean nameCanBeVisited(SimpleName id) {
            String name = id.asString();
            return p.matcher(name).matches();
        }

        public T visitTypeName(SimpleName id) {
            return null;
        }
    }

    /***
     * A Visitor that locates all the ImportDeclarations in a file and adds them all to a list
     */
    private static class Replacer extends VoidVisitorAdapter<Void> {

        public void visit(IntegerLiteralExpr id, Void arg) {
           super.visit(id, arg);
           setDefault(id, "<INT>", new String[] {"0", "1", "2"});
        }

        public void visit(DoubleLiteralExpr id, Void arg) {
            super.visit(id, arg);
            setDefault(id, "<DOUBLE>", new String[] {"0", ".0", "0."});
        }

        public void visit(StringLiteralExpr id, Void arg) {
            super.visit(id, arg);
            setDefault(id, "<STRING>", null);
        }

        public void visit(CharLiteralExpr id, Void arg) {
            super.visit(id, arg);
            setDefault(id, "<CHAR>", null);
        }

        public void visit(LongLiteralExpr id, Void arg) {
            super.visit(id, arg);
            setDefault(id, "<LONG>", new String[] {"0", "1", "0L", "1L"});
        }

        public <T extends LiteralStringValueExpr> void setDefault(T id, String defaultValue, String[] keep) {
            assert(id.getTokenRange().isPresent());
            JavaToken token = id.getTokenRange().get().getBegin();
            assert(token == id.getTokenRange().get().getEnd());
            if (keep != null)
                for (String value: keep)
                    if (value.equals(token.getText()))
                        return;
            token.setText(defaultValue);
            id.setValue(defaultValue);
        }
    }
}
