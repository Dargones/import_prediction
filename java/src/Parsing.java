import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.MarkerAnnotationExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.type.ArrayType;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.type.WildcardType;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
//import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;

public class Parsing {
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

    public static String visitAll(String file) {
        //CompilationUnit tree = JavaParser.parse(file);
        //return visitOne(tree);
        return null;
    }

    public static String visitOne(Node node) {
        String result = "";
        result += node.toString();
        for (Node child: node.getChildNodes()) {
            result += visitOne(child);
        }
        return result;
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
        //static ChiSquareTest chi = new ChiSquareTest(); // for performing the test
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

            //this.pValue = chi.chiSquare(new long[][] {{cooc, only0}, {only1, none}});
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

    /***
     * A Visitor that locates all the ImportDeclarations in a file and adds them all to a list
     */
    private static class ImportNamePrinter extends VoidVisitorAdapter<List<String>> {
        static Pattern p = Pattern.compile("[A-Z][A-Za-z0-9]*");

        @Override
        public void visit(ClassOrInterfaceType id, List<String> names) {
            super.visit(id, names);  // not necessary, but documentation suggests using it anyway
            recursiveVisit(id, names);
        }

        public void recursiveVisit(Node id, List<String> names) {
            for (Node node: id.getChildNodes())
                if ((node instanceof ClassOrInterfaceType) || (node instanceof ArrayType)) {
                    recursiveVisit(node, names);
                } else if (node instanceof SimpleName) {
                    String name = ((SimpleName)node).asString();
                    if (p.matcher(name).matches())
                        names.add(name);
                } else if ((node instanceof WildcardType) ||
                        (node instanceof PrimitiveType) ||
                        (node instanceof MarkerAnnotationExpr) ||
                        (node instanceof SingleMemberAnnotationExpr)) {

                } else {
                    System.out.println("Problem here");
                }
        }
    }
}