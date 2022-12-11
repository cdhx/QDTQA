package cn.edu.nju.ws.edgqa.domain.edg;

import cn.edu.nju.ws.edgqa.domain.beans.tuple.ThreeTuple;
import cn.edu.nju.ws.edgqa.utils.UriUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.QueryType;
import cn.edu.nju.ws.edgqa.utils.enumerates.Trigger;
import cn.edu.nju.ws.edgqa.utils.kbutil.KBUtil;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.Syntax;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.stream.Collectors;

public class SparqlGenerator implements Comparable {

    /**
     * The type of this question
     */
    private QueryType quesType;

    /**
     * the final variable name without the question mark
     */
    private String finalVarName;

    /**
     * A list of sparql variables
     */
    private List<String> varList;

    /**
     * A list of sparql triples
     */
    private List<ThreeTuple<String, String, String>> tupleList;
    private HashMap<String, String> varValueMap; // <e0,uri> for judge，replace ?e0... with explicit entity uri
    private Double score; // the confidence score of this SparqlGenerator

    private Trigger trigger = Trigger.UNKNOWN;

    /**
     * Constructor
     */
    public SparqlGenerator() {
        this.quesType = QueryType.UNKNOWN;
        this.finalVarName = "";
        this.varList = new ArrayList<>();
        this.tupleList = new ArrayList<>();
        this.varValueMap = new HashMap<>();
        this.score = 1.0;
    }

    /**
     * Constructor
     *
     * @param quesType the type of this question
     */
    public SparqlGenerator(QueryType quesType) {
        this.quesType = quesType;
        this.finalVarName = "";
        this.varList = new ArrayList<>();
        this.tupleList = new ArrayList<>();
        this.varValueMap = new HashMap<>();
        this.score = 1.0;
    }

    /**
     * Constructor
     *
     * @param quesType  the type of this question
     * @param varList   the list of variables
     * @param tupleList the list of 3-tuples
     */
    public SparqlGenerator(QueryType quesType, String finalVarName, List<String> varList, List<ThreeTuple<String, String, String>> tupleList) {
        this.quesType = quesType;
        this.finalVarName = finalVarName;
        this.varList = varList;
        this.tupleList = tupleList;
        this.varValueMap = new HashMap<>();
        this.score = 1.0;
    }

    /**
     * Copy constructor
     *
     * @param sparqlGenerator another instance of class SparqlGenerator
     */
    public SparqlGenerator(SparqlGenerator sparqlGenerator) {
        this.quesType = sparqlGenerator.getQuesType();
        this.finalVarName = sparqlGenerator.getFinalVarName();
        this.varList = new ArrayList<>();
        this.tupleList = new ArrayList<>();
        this.varValueMap = new HashMap<>();
        this.score = sparqlGenerator.getScore();

        if (!sparqlGenerator.getVarList().isEmpty())
            this.varList.addAll(sparqlGenerator.getVarList());
        if (!sparqlGenerator.getTupleList().isEmpty())
            this.tupleList.addAll(sparqlGenerator.getTupleList());
        if (!sparqlGenerator.getVarValueMap().isEmpty())
            this.varValueMap.putAll(sparqlGenerator.getVarValueMap());
    }

    public Trigger getTrigger() {
        return trigger;
    }

    public void setTrigger(Trigger trigger) {
        this.trigger = trigger;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public QueryType getQuesType() {
        return quesType;
    }

    public void setQuesType(QueryType quesType) {
        this.quesType = quesType;
    }

    public String getFinalVarName() {
        return finalVarName;
    }

    public void setFinalVarName(String finalVarName) {
        this.finalVarName = finalVarName;
        // modify the varList simultaneously
        varList.add(finalVarName);

        HashSet<String> tempSet = new HashSet<>(varList);
        varList.clear();
        varList.addAll(tempSet);
    }

    public List<String> getVarList() {
        return varList;
    }

    public void setVarList(List<String> varList) {
        this.varList = varList;
    }

    public List<ThreeTuple<String, String, String>> getTupleList() {
        return tupleList;
    }

    public void setTupleList(List<ThreeTuple<String, String, String>> tupleList) {
        this.tupleList = tupleList;
    }

    public HashMap<String, String> getVarValueMap() {
        return varValueMap;
    }

    public void setVarValueMap(HashMap<String, String> varValueMap) {
        this.varValueMap = varValueMap;
    }

    /**
     * Add a triple to the tupleList.
     * The subject and object are distinguished, you don't have to flip the two and add them again.
     *
     * @param subject   subject of a triple
     * @param predicate predicate of a triple
     * @param object    object of a triple
     */
    public void addTriple(String subject, String predicate, String object) {
        addTriple(new ThreeTuple<>(subject, predicate, object));
    }

    public String addAngleBrackets(String str) {
        str = str.trim();
        // variable, just return
        if (str.startsWith("?"))
            return str;
        // not a variable but a URI
        if (!str.startsWith("<"))
            str = "<" + str;
        if (!str.endsWith(">"))
            str = str + ">";
        return str;
    }

    /**
     * add a ThreeTuple to the tupleList
     *
     * @param threeTuple the threeTuple to be added
     */
    public void addTriple(ThreeTuple<String, String, String> threeTuple) {
        threeTuple = new ThreeTuple<>(addAngleBrackets(threeTuple.getFirst()),
                addAngleBrackets(threeTuple.getSecond()),
                addAngleBrackets(threeTuple.getThird()));
        tupleList.add(threeTuple);

        // add elements to varList
        if (threeTuple.getFirst().matches("^\\?(.*)$")) {
            varList.add(threeTuple.getFirst().replace("?", "").trim());
        }
        if (threeTuple.getSecond().matches("^\\?(.*)$")) {
            varList.add(threeTuple.getSecond().replace("?", "").trim());
        }
        if (threeTuple.getThird().matches("^\\?(.*)$")) {
            varList.add(threeTuple.getThird().replace("?", "").trim());
        }

        varList = varList.stream().distinct().collect(Collectors.toList());
    }

    public void addTupleList(List<ThreeTuple<String, String, Double>> threeTupleList, String variable) {

        for (ThreeTuple<String, String, Double> threeTuple : threeTupleList) {
            tupleList.add(new ThreeTuple<>(variable, "<" + threeTuple.getSecond() + ">", "<" + threeTuple.getFirst() + ">"));
            //this.score*=threeTuple.getThird();
        }

        varList.add(variable.replace("?", "").trim());
        varList = varList.stream().distinct().collect(Collectors.toList());

    }

    /**
     * Convert the tuples to a sparql query.
     * Lc-Quad training data only contains three types of SPARQL queries:
     * 1. SELECT DISTINCT ?uri WHERE ...
     * 2. SELECT DISTINCT COUNT(?uri) WHERE ...
     * 3. ASK WHERE ...
     * Note that solving the counting problem may require looking up properties in the KB first.
     * The strategy for judge question is not generating a "ASK" query.
     *
     * @return a sparql string
     */
    public String toSparql() {
        String finalVar = finalVarName;
        StringBuilder query = new StringBuilder();
        if (quesType == QueryType.COUNT) {

            // detection of numerical attributes
            SparqlGenerator temp = new SparqlGenerator(this);
            temp.setQuesType(QueryType.LIST);
            List<String> tempResult = KBUtil.getQueryStringResult(temp.toQuery());
            if (tempResult.size() == 1 && tempResult.get(0).matches("(\\d+)|(\\d+.e-\\d+)")) {
                query.append("SELECT ").append("?").append(finalVar).append(" WHERE { ");
                this.setQuesType(QueryType.LIST);
            } else {
                query.append("SELECT ").append("(COUNT(DISTINCT ?").append(finalVar).append(") AS ?ans) WHERE { ");
            }

        } else if (quesType == QueryType.JUDGE) {
            query.append("ASK WHERE { ");
        } else {
            query.append("SELECT DISTINCT ").append("?").append(finalVar).append(" WHERE { ");
        }
        query.append(tupleListToPartialSparql());
        query.append("}");
        return query.toString();
    }

    /**
     * Convert the tuple list to a substring of the query
     *
     * @return a substring of the query
     */
    public String tupleListToPartialSparql() {
        StringBuilder query = new StringBuilder();

        for (ThreeTuple<String, String, String> tuple : tupleList) {
            query.append("{").append(tuple.getFirst()).append(" ")
                    .append(tuple.getSecond()).append(" ")
                    .append(tuple.getThird()).append("}");

            // if not type, bi-direction
            if (!tuple.getSecond().equals("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
                query.append(" UNION ");
                query.append("{").append(tuple.getThird()).append(" ").
                        append(tuple.getSecond()).append(" ").
                        append(tuple.getFirst()).append("}").append(" . ");
            } else {
                query.append(" .");
            }
        }

        String queryStr = query.toString();
        if (!varValueMap.keySet().isEmpty()) {
            for (String var : varValueMap.keySet()) {
                queryStr = queryStr.replaceAll("\\?" + var, varValueMap.get(var));
            }
        }

        return queryStr;
    }

    /**
     * Convert the sparql string to a query
     *
     * @return a sparql Query
     */
    public Query toQuery() {
        return QueryFactory.create(toSparql(), Syntax.syntaxARQ);
    }

    /**
     * Execute this sparql.
     *
     * @return a list of QuerySolution
     */
    public ArrayList<QuerySolution> execute() {
        return KBUtil.runQuery(toQuery());
    }

    /**
     * Get the result of this sparql
     *
     * @return answer list, no duplicate elements
     */
    public List<String> solve() {

        List<String> answerList = new ArrayList<>();

        ArrayList<QuerySolution> querySolutions = KBUtil.runQuery(this.toQuery()); // query DBpedia
        if (querySolutions.isEmpty() && this.getQuesType() != QueryType.JUDGE)
            return answerList;
        if (this.getQuesType() == QueryType.JUDGE) {
            if (!querySolutions.isEmpty()) {
                boolean answer = querySolutions.get(0).getLiteral("__ask_retval").toString().equals("1^^http://www.w3.org/2001/XMLSchema#integer");
                answerList.add(String.valueOf(answer));
            } else {
                answerList.add("false");
            }
        } else if (this.getQuesType() == QueryType.COUNT) {
            String[] strArr;
            strArr = querySolutions.get(0).getLiteral("ans").toString().split("\\^\\^");
            int resNum = Integer.parseInt(strArr[0]);
            answerList.add(String.valueOf(resNum));
        } else {  // COMMON type
            for (QuerySolution qs : querySolutions) {
                Iterator<String> var = qs.varNames();
                while (var.hasNext()) {
                    String varName = var.next();
                    if (varName.equals(finalVarName)) {
                        String answer = qs.get(varName).toString();

                        if (answer.contains("^^")) {
                            answer = answer.split("\\^\\^")[0];
                        }
                        answerList.add(answer);
                    }
                }
            }
        }

        return answerList.stream().distinct().collect(Collectors.toList());
    }

    /**
     * expand the query by flip 'http://dbpedia.org/ontology' to 'http://dbpedia.org/property'
     *
     * @return expanded queryList
     */
    public List<SparqlGenerator> expandQueryWithDbpOrDbo() {

        List<SparqlGenerator> result = new ArrayList<>();
        SparqlGenerator newSPG = new SparqlGenerator(this);
        newSPG.tupleList.clear();
        result.add(newSPG);

        // for each tuple in tupleList, expand it
        for (ThreeTuple<String, String, String> threeTuple : tupleList) {
            String second = threeTuple.getSecond();
            String label = UriUtil.splitLabelFromUri(second);

            ThreeTuple<String, String, String> newTuple = new ThreeTuple<>(threeTuple);

            if (second.startsWith("<http://dbpedia.org/ontology/") && Character.isLowerCase(label.charAt(0))) {
                String newPredicate = second.replace("http://dbpedia.org/ontology/", "http://dbpedia.org/property/");
                newTuple.setSecond(newPredicate);
            } else if (second.startsWith("<http://dbpedia.org/property/") && Character.isLowerCase(label.charAt(0))) {
                String newPredicate = second.replace("http://dbpedia.org/property/", "http://dbpedia.org/ontology/");
                newTuple.setSecond(newPredicate);
            }

            List<SparqlGenerator> immediateList = new ArrayList<>();
            for (SparqlGenerator spg : result) {

                SparqlGenerator tempSPG = new SparqlGenerator(spg);

                // origin tuple
                spg.addTriple(threeTuple);
                immediateList.add(spg);

                //new tuple
                if (!newTuple.equals(threeTuple)) {
                    tempSPG.addTriple(newTuple);
                    immediateList.add(tempSPG);
                }

            }
            result = immediateList;
        }

        // remove empty query
        result.removeIf(sp -> KBUtil.isZeroSelect(sp.toString()));
        // remove duplicate
        result = result.stream().distinct().collect(Collectors.toList());

        return result;

    }

    /**
     * set the finalVarName to a new VarName
     *
     * @param newVarName new VarName
     */
    public void changeFinalVarName(String newVarName) {
        String oldVarName = getFinalVarName();
        setFinalVarName(newVarName);

        varList.remove(oldVarName);
        varList.add(newVarName);

        if (varValueMap.containsKey(oldVarName)) {
            varValueMap.put(newVarName, varValueMap.get(oldVarName));
        }

        tupleList.forEach(tuple -> {
            if (tuple.getFirst().equals("?" + oldVarName)) {
                tuple.setFirst("?" + newVarName);
            }
            if (tuple.getSecond().equals("?" + oldVarName)) {
                tuple.setSecond("?" + newVarName);
            }
            if (tuple.getThird().equals("?" + oldVarName)) {
                tuple.setThird("?" + newVarName);
            }
        });

    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SparqlGenerator that = (SparqlGenerator) o;
        return toSparql().equals(that.toSparql());
    }

    @Override
    public int hashCode() {
        return Objects.hash(toSparql());
    }

    @Override
    public String toString() {
        return this.toSparql();
    }

    public String tupleListToString_QALD() {
        StringBuilder sb = new StringBuilder();
        for (int h = 0; h < tupleList.size(); h++) {
            sb.append(" [TRP] ");
            ThreeTuple<String, String, String> triple = tupleList.get(h);

            String subject = triple.getFirst().replace("<", "").replace(">", "");
            String subjectLabel = subject;
            if (subject.contains("http://")) {
                /* ojbect is a uri */
                String label = KBUtil.queryLabel(subject);
                if (label != null) {
                    subjectLabel = label;
                } else {
                    subjectLabel = UriUtil.extractUri(subject);
                }
            }
            sb.append(subjectLabel).append(" ");


            String predicate = triple.getSecond().replace("<", "").replace(">", "");
            String predicateLabel = predicate;
            if (predicate.contains("http://")) {
                /* predicate is a uri*/
                if (predicate.equals("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")) {
                    predicateLabel = "type";
                } else {
                    String label = KBUtil.queryLabel(predicate);
                    if (label != null) {
                        predicateLabel = label;
                    } else {
                        predicateLabel = UriUtil.extractUri(predicate);
                    }
                }

            }
            sb.append(predicateLabel).append(" ");

            String object = triple.getThird().replace("<", "").replace(">", "");
            String objectLabel = object;
            if (object.contains("http://")) {
                /* ojbect is a uri */
                String label = KBUtil.queryLabel(object);
                if (label != null) {
                    objectLabel = label;
                } else {
                    objectLabel = UriUtil.extractUri(object);
                }
            }
            sb.append(objectLabel);

        }

        sb.insert(0, quesType);
        return sb.toString();
    }

    public String tupleListToString_LCQUAD() {
        StringBuilder sb = new StringBuilder();
        for (int h = 0; h < tupleList.size(); h++) {
            sb.append(" [TRP] ");
            ThreeTuple<String, String, String> triple = tupleList.get(h);

            String subject = triple.getFirst();
            sb.append(subject).append(" ");


            String predicate = triple.getSecond().replace("<", "").replace(">", "");
            String predicateLabel = predicate;
            if (predicate.contains("http://")) {
                // predicate is a uri
                if (predicate.equals("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")) {
                    predicateLabel = "type";
                } else {
                    String label = KBUtil.queryLabel(predicate);
                    if (label != null) {
                        predicateLabel = label;
                    } else {
                        predicateLabel = UriUtil.extractUri(predicate);
                    }
                }

            }
            sb.append(predicateLabel).append(" ");

            String object = triple.getThird().replace("<", "").replace(">", "");
            String objectLabel = object;
            if (object.contains("http://")) {
                // ojbect is a uri
                String label = KBUtil.queryLabel(object);
                if (label != null) {
                    objectLabel = label;
                } else {
                    objectLabel = UriUtil.extractUri(object);
                }
            }
            sb.append(objectLabel);

        }
        return sb.toString();
    }


    @Override
    public int compareTo(@NotNull Object o) {

        if (!(o instanceof SparqlGenerator)) {
            return 1;
        } else {
            return Double.compare(this.score, ((SparqlGenerator) o).getScore());
        }

    }

    public boolean contains(SparqlGenerator other) {
        Set<String> entitySet = new HashSet<>();
        Set<String> relationSet = new HashSet<>();

        for (ThreeTuple<String, String, String> threeTuple : tupleList) {
            entitySet.add(threeTuple.getFirst());
            entitySet.add(threeTuple.getThird());
            relationSet.add(threeTuple.getSecond());
        }

        for (ThreeTuple<String, String, String> threeTuple : other.tupleList) {
            if (!entitySet.contains(threeTuple.getFirst())) return false;
            if (!entitySet.contains(threeTuple.getThird())) return false;
            if (!relationSet.contains(threeTuple.getSecond())) return false;
        }
        return true;
    }
}
