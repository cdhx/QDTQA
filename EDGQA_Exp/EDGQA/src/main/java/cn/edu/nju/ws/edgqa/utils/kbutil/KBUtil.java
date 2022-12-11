package cn.edu.nju.ws.edgqa.utils.kbutil;

import cn.edu.nju.ws.edgqa.domain.beans.Link;
import cn.edu.nju.ws.edgqa.domain.beans.tuple.ThreeTuple;
import cn.edu.nju.ws.edgqa.handler.Detector;
import cn.edu.nju.ws.edgqa.utils.UriUtil;
import cn.edu.nju.ws.edgqa.utils.enumerates.DatasetEnum;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Literal;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.ResourceFactory;
import virtuoso.jena.driver.VirtGraph;
import virtuoso.jena.driver.VirtuosoQueryExecution;
import virtuoso.jena.driver.VirtuosoQueryExecutionFactory;

import java.util.*;

/**
 * Tools for querying DBpedia
 */
public class KBUtil {
    private static final String CORE_KB_1604 = "http://210.28.134.34:8892/";
    private static final String CORE_KB_GRAPH_1604 = "http://dbpedia.org/";
    private static final String CORE_KB_JDBC_1604 = "jdbc:virtuoso://210.28.134.34:1113";
    private static final String CORE_KB_1610 = "http://210.28.134.34:8891/";
    private static final String CORE_KB_GRAPH_1610 = "http://dbpedia.org";
    private static final String CORE_KB_JDBC_1610 = "jdbc:virtuoso://210.28.134.34:1112";
    // use dbpedia 1604 by default
    private static final String CORE_KB_FREEBASE = "http://210.28.134.34:8890/";
    private static final String CORE_KB_GRAPH_FREEBASE = "http://freebase.com";
    private static final String CORE_KB_JDBC_FREEBASE = "jdbc:virtuoso://210.28.134.34:1111";
    public static VirtGraph connection = null;
    public static DatasetEnum dataset = DatasetEnum.UNKNOWN; // by default, 1604
    // use dbpedia 1604 by default
    private static String CORE_KB = CORE_KB_1604;
    private static String CORE_KB_GRAPH = CORE_KB_GRAPH_1604;
    private static String CORE_KB_JDBC = CORE_KB_JDBC_1604;


    // init DataSet
    public static void init(DatasetEnum dataset) {

        System.out.println("[INFO] KBUtil initializing...");
        // choose kb by dataset
        switch (dataset) {
            case QALD_9: {
                CORE_KB = CORE_KB_1610;
                CORE_KB_GRAPH = CORE_KB_GRAPH_1610;
                CORE_KB_JDBC = CORE_KB_JDBC_1610;
                break;
            }
            case LC_QUAD:
            default: {
                CORE_KB = CORE_KB_1604;
                CORE_KB_GRAPH = CORE_KB_GRAPH_1604;
                CORE_KB_JDBC = CORE_KB_JDBC_1604;
                break;
            }
        }
        // establish connection
        connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");
        System.out.println("[INFO] KBUtil initialized");
    }

    /**
     * test whether a clause exists
     *
     * @param clause a sparql clause in where
     * @return true for exists; false for not
     */
    public static boolean isClauseExist(String clause) {
        String queryString = "ASK WHERE {" + clause + "}";
        return runAskOnVirtuoso(queryString);
    }

    //Determine whether a select count result is 0
    public static boolean isZeroSelect(String queryString) {
        ArrayList<QuerySolution> results = runSelectOnVirtuoso(queryString);
        if (results.size() == 0)
            return true;
        QuerySolution qs = results.get(0);
        for (Iterator<String> it = qs.varNames(); it.hasNext(); ) {
            String varName = it.next();
            if (qs.get(varName).isLiteral() && qs.get(varName).asLiteral().getString().equals("0"))
                return true;
        }
        return false;
    }

    //Determine whether a query result is an empty set
    public static boolean isEmptySelect(String queryString) {
        //			System.out.println(runSelectOnVirtuoso(queryString));
        return runSelectOnVirtuoso(queryString).size() == 0;
    }

    //Determine whether there is any <? x, s, p> x
    public static boolean isPOSatisfiable(String p, String o) {
        String query = "ASK WHERE {?x " + p + " " + o + "}";
        return runAskOnVirtuoso(query);
    }

    //Determine whether there is any <s,p,?x> x
    public static boolean isSPSatisfiable(String s, String p) {
        String query = "ASK WHERE {" + s + " " + p + " ?x }";
        return runAskOnVirtuoso(query);
    }

    // Determine whether there is <s, ?x, o> x is any relation
    public static boolean isSOSatisfiable(String s, String o) {
        String query = "ASK WHERE { " + s + " ?x " + o + "}";
        return runAskOnVirtuoso(query);
    }

    //Judge whether <s,p,o> is true
    public static boolean isSPOSatisfiable(String s, String p, String o) {
        String query = "ASK WHERE {" + s + " " + p + " " + o + "}";
        return runAskOnVirtuoso(query);
    }

    //Determine whether <e,r,?> or <?,e,r> exists
    public static boolean isERexists(String eURI, String rURI) {
        String query = "ASK WHERE { {<" + eURI + "> <" + rURI + "> ?p} UNION {?p <" + rURI + "> <" + eURI + ">}.}";
        //System.out.println(query);
        return runAskOnVirtuoso(query);

    }

    //Determine whether <e1,r,e2> or <e2,r,e1> exists
    public static boolean isE1RE2exists(String eURI1, String rURI, String eURI2) {
        String query = "ASK WHERE { {<" + eURI1 + "> <" + rURI + "> <" + eURI2 + ">} UNION {<" + eURI2 + "> <" + rURI + "> <" + eURI1 + ">}.}";
        //System.out.println(query);
        return runAskOnVirtuoso(query);
    }

    //Determine whether the query corresponding to a tupleList has a result
    public static boolean isThreeTupleListExists(List<ThreeTuple<String, String, Double>> tupleList) {

        String query = "ASK WHERE {";
        for (ThreeTuple<String, String, Double> tuple : tupleList) {
            String eURI = tuple.getFirst();
            String rURI = tuple.getSecond();
            query = query + " {<" + eURI + "> <" + rURI + "> ?x} UNION {?x <" + rURI + "> <" + eURI + ">}.\n";
        }
        query += "}";
        return runAskOnVirtuoso(query);

    }

    //Determine whether the two entities are within two hops
    public static boolean isEPairInTwoHop(String eURI1, String eURI2) {

        // in OneHop
        if (isEPairInOneHop(eURI1, eURI2)) {
            return true;
        }

        String query = "ASK WHERE {{<" + eURI1 + "> ?p1 ?x" + "} UNION {?x ?p1 <" + eURI1 + ">}.\n"
                + "{<" + eURI2 + "> ?p2 ?x" + "} UNION {?x ?p2 <" + eURI2 + ">}.}\n";
        //+ "Filter(?p1 != rdf:type && ?p2 != rdf:type).\n }\n";

        //System.out.println("query:" + query);

        return runAskOnVirtuoso(query);

    }

    //Determine whether two entities are adjacent
    public static boolean isEPairInOneHop(String eURI1, String eURI2) {

        String query = "ASK WHERE { {<" + eURI1 + "> ?p <" + eURI2 + ">} UNION {<" + eURI2 + "> ?p <" + eURI1 + ">}.}";
        return runAskOnVirtuoso(query);

    }

    // Execute select query and return the answer collection
    public static ArrayList<QuerySolution> runSelectOnVirtuoso(String queryString) {
        if (KBUtil.connection == null)
            //DBpediaTools.connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");
            connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");
        if (queryString.contains("COUNT"))                            //Sometimes distinct + count goes wrong (Jena), strange!
            queryString = queryString.replaceAll("SELECT DISTINCT", "SELECT");
        Query sparql = QueryFactory.create(queryString, Syntax.syntaxARQ);
        if (sparql.getLimit() == 0) {
            sparql.setLimit(100);
        }
        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(sparql, connection);
        ArrayList<QuerySolution> ret = new ArrayList<QuerySolution>();
        try {
            vqe.execSelect().forEachRemaining(x -> ret.add(x));
            //QuerySolution s = vqe.execSelect().next();
            //String x = s.get("?uri").toString();
            //System.out.println("fewf"+x);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            vqe.close();
        }
        return ret;
    }

    // Execute ask query, return yes/no
    public static boolean runAskOnVirtuoso(String queryString) {
        if (KBUtil.connection == null)
            KBUtil.connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");

        queryString = addPrefix(queryString);
        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(queryString, connection);
        boolean re = false;
        try {
            re = vqe.execAsk();
        } finally {
            vqe.close();
        }
        return re;
    }

    /**
     * Execute the select query
     *
     * @param query the KB query
     * @return the answer collection
     */
    public static ArrayList<QuerySolution> runQuery(Query query) {
        if (KBUtil.connection == null)
            KBUtil.connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");


        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(query, connection);
        ArrayList<QuerySolution> ret = new ArrayList<QuerySolution>();
        try {
            vqe.execSelect().forEachRemaining(ret::add);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            vqe.close();
        }
        return ret;
    }

    // Execute all types of queries and return List<String>
    public static List<String> getQueryStringResult(Query query) {

        List<String> answerList = new ArrayList<>();

        if (query.isAskType()) {//ask
            answerList.add(String.valueOf(runAskOnVirtuoso(query.toString())));
            return answerList;
        }

        //not ask
        ArrayList<QuerySolution> querySolutions = runQuery(query);
        //System.out.println(querySolutions.size());
        for (QuerySolution querySolution : querySolutions) {
            Iterator<String> it = querySolution.varNames();
            while (it.hasNext()) {
                String s = it.next();
                if (!s.equals("graph")) {
                    if (s.startsWith("callret")) {
                        answerList.add(querySolution.get(s).toString().split("\\^\\^")[0]);
                    } else if (s.startsWith("__ask_retval")) {
                        if (querySolution.get(s).toString().split("\\^\\^")[0].equals("1")) {
                            answerList.add("true");
                        } else {
                            answerList.add("false");
                        }
                    } else {

                        String answer = querySolution.get(s).toString();
                        if (answer.contains("^^")) {//has type
                            answer = answer.split("\\^\\^")[0];
                        }

                        answerList.add(answer);
                    }
                }
            }
        }
        return answerList;
    }

    // Add a prefix, such as dbo: <http://dbpedia.org/ontology/>
    public static String addPrefix(String originQuery) {
        /* The prefix of QALD */
        String prefix = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                "PREFIX dbp: <http://dbpedia.org/property/>\n" +
                "PREFIX res: <http://dbpedia.org/resource/>\n" +
                "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n" +
                "PREFIX dbo: <http://dbpedia.org/ontology/>\n" +
                "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" +
                "PREFIX dct: <http://purl.org/dc/terms/>\n" +
                "PREFIX dbc: <http://dbpedia.org/resource/Category:>\n" +
                "PREFIX yago: <http://dbpedia.org/class/yago/>\n" +
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n" +
                "PREFIX dbr: <http://dbpedia.org/resource/>\n" +
                "PREFIX ns: <http://rdf.freebase.com/ns/>\n";
        return prefix + originQuery;
    }

    /**
     * Get the one-hop properties of an entity
     *
     * @param entityURI the entity URI
     * @return the one hop properties of this entity
     */
    public static Set<String> oneHopProperty(String entityURI) {
        String sparql = ""
                + "select ?p where {\n"
                + "  {?x ?p <" + entityURI + ">}\n"
                + "union\n"
                + "{<" + entityURI + "> ?p ?y}"
                + "}";
        sparql = addPrefix(sparql);

        if (KBUtil.connection == null)
            connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");

        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(sparql, connection);
        ArrayList<QuerySolution> ret = new ArrayList<QuerySolution>();
        try {
            vqe.execSelect().forEachRemaining(ret::add);
        } catch (Exception e) {
            System.out.println(sparql);
            e.printStackTrace();
        } finally {
            vqe.close();
        }
        HashSet<String> result = new HashSet<>();
        for (QuerySolution qsol : ret) {
            result.add(qsol.get("?p").toString());

        }
        return result;
    }

    /**
     * Get two-hop types
     *
     * @param entityURI the entity URI
     * @return the two hop types of this entity
     */
    public static Set<String> twoHopTypes(String entityURI) {
        String query = ""
                + "select ?type where {\n"
                + "{{?e ?p ?x}\n"
                + "union\n"
                + "{?x ?p ?e}}.\n"
                + "?x rdf:type ?type.\n"
                + "}";
        query = addPrefix(query);
        ParameterizedSparqlString qs = new ParameterizedSparqlString(query);

        qs.setIri("e", entityURI);
        //System.out.println("query:" + qs.toString());

        if (KBUtil.connection == null)
            //DBpediaTools.connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");
            connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");

        Query sparql = qs.asQuery();

        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(sparql, connection);
        ArrayList<QuerySolution> ret = new ArrayList<QuerySolution>();
        try {
            vqe.execSelect().forEachRemaining(x -> ret.add(x));
        } catch (Exception e) {
            System.out.println(qs);
            e.printStackTrace();
        } finally {
            vqe.close();
        }
        HashSet<String> result = new HashSet<>();
        for (QuerySolution qsol : ret) {
            result.add(qsol.get("?type").toString());

        }

        result.removeIf(uri -> !uri.startsWith("http://dbpedia.org/ontology"));

        return result;
    }

    // Check label according to uri
    public static String queryLabel(String uri) {


        String sparql = "select ?x where {\n" + "  {<" + uri + "> rdfs:label ?x}\n" + "}";
        sparql = addPrefix(sparql);

        if (KBUtil.connection == null)
            connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");

        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(sparql, connection);
        ArrayList<QuerySolution> ret = new ArrayList<QuerySolution>();
        try {
            vqe.execSelect().forEachRemaining(ret::add);
        } catch (Exception e) {
            System.out.println(sparql);
            e.printStackTrace();
        } finally {
            vqe.close();
        }
        List<String> result = new ArrayList<>();
        for (QuerySolution qsol : ret) {
            result.add(qsol.get("?x").toString());

        }
        if (result.size() > 0) {
            for (String str : result) {
                if (str.endsWith("@en")) {
                    return str.replace("@en", "");
                }
            }
        }
        return null;
    }

    // Check uri according to wikiPageID
    public static String queryByID(int id) {
        String result = null;

        if (KBUtil.connection == null)
            KBUtil.connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");

        //for DBpedia
        if (CORE_KB_GRAPH == CORE_KB_GRAPH_FREEBASE) {
            String sparqlString = "SELECT ?resource where{\n" +
                    "?resource <http://rdf.freebase.com/key/wikipedia.en_id> \"" + id + "\".}";
            sparqlString = addPrefix(sparqlString);
            //System.out.println(sparqlString);
            Query query = QueryFactory.create(sparqlString);
            List<String> queryStringResult = getQueryStringResult(query);
            if (!queryStringResult.isEmpty()) {
                return queryStringResult.get(0);
            } else {
                return null;
            }

        } else {
            String sparqlString = ""
                    + "select ?resource where {\n"
                    + "  ?resource ?p ?id.\n"
                    + "}";
            //System.out.println(sparqlString);
            try {
                //Query query = QueryFactory.create(sparqlString, Syntax.syntaxARQ);
                ParameterizedSparqlString qs = new ParameterizedSparqlString(sparqlString);

                Property property = ResourceFactory.createProperty("http://dbpedia.org/ontology/", "wikiPageID");
                qs.setParam("p", property);

                Literal wikiID = ResourceFactory.createTypedLiteral(id);
                qs.setParam("id", wikiID);

                Query query = QueryFactory.create(qs.toString());
                //System.out.println(query.toString());

                VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(query, connection);

                try {
                    ResultSet resultSet = vqe.execSelect();
                    if (resultSet.hasNext()) {
                        result = resultSet.next().get("resource").toString();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    vqe.close();
                }


            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    public static Set<String> getOneHopPotentialEntity(List<Link> linkList) {
        Set<String> res = new HashSet<>();
        for (Link link : linkList) {
            if (link.getScore() >= 0.90) {
                Set<String> oneHop = KBUtil.oneHopProperty(link.getUri());
                Set<String> oneHopLabel = new HashSet<>();
                for (String str : oneHop) {
                    str = UriUtil.extractUri(str);
                    if (str.contains("wiki") && str.contains("page")) continue;
                    if (str.equals("22-rdf-syntax-ns#type") || str.contains("rdf-schema#") || str.contains("owl#"))
                        continue;
                    if (str.length() < 3) continue; // too short label
                    oneHopLabel.add(str);
                }
                res.addAll(oneHopLabel);
            }
        }
        return res;
    }

    /**
     * judge whether a uri is a Title, such as "President of USA" ...
     *
     * @param uri uri
     * @return true if uri is a title, false o.w.
     */
    public static boolean isaTitle(String uri) {

        // not xx of xx
        if (!UriUtil.extractUri(uri).matches("(.*) of (.*)")) {
            return false;
        }

        String sparqlString = "ASK WHERE{{?x dbp:title <" + uri + ">} UNION {?x dbo:title <" + uri + ">}.}";
        return runAskOnVirtuoso(sparqlString);
    }

    /**
     * judge whether a uri is a disambiguation Page, such as "dbr:Neil_Brown"
     *
     * @param uri uri
     * @return true if uri is a disambiguation page, false o.w.
     */
    public static boolean isDisambiguationPage(String uri) {
        String sparqlString = "ASK WHERE{<" + uri + "> dbo:wikiPageDisambiguates ?x}";
        return runAskOnVirtuoso(sparqlString);
    }

    /**
     * judge whether a uri is a redirect Page, such as "Paris_mayors"
     *
     * @param uri uri
     * @return true for is a redirect page, false o.w.
     */
    public static boolean isRedirectPage(String uri) {
        String sparqlString = "ASK WHERE{<" + uri + "> dbo:wikiPageRedirects ?x}";
        return runAskOnVirtuoso(sparqlString);
    }

    /**
     * Determine whether a uri is a dbpedia entity
     *
     * @param uri Possible uri
     * @return true for is a dbpedia instance
     */
    public static boolean isAnInstance(String uri) {
        String sparqlString = "ASK WHERE{<" + uri + "> rdf:type ?x}";
        return runAskOnVirtuoso(sparqlString);
    }

    /**
     * if a uri is a title, get the instances that have this title
     *
     * @param uri title uri
     * @return instancesUri
     */
    public static List<String> getTitleInstances(String uri) {
        String sparqlString = "SELECT ?x WHERE{{?x dbp:title <" + uri + ">} UNION {?x dbo:title <" + uri + ">}.}";
        //System.out.println(sparqlString);
        sparqlString = addPrefix(sparqlString);
        return getQueryStringResult(QueryFactory.create(sparqlString));
    }

    public static List<String> getRedirectPage(String uri) {
        String sparqlString = "SELECT ?x WHERE{<" + uri + "> dbo:wikiPageRedirects ?x}";
        sparqlString = addPrefix(sparqlString);
        return getQueryStringResult(QueryFactory.create(sparqlString));
    }

    /**
     * give the result of a potential ER tuple
     *
     * @param eUri eURI
     * @param rUri rURI
     * @return query result
     */
    public static List<String> getERRes(String eUri, String rUri) {
        String query = "SELECT ?x WHERE{ { <" + eUri + "> <" + rUri + "> ?x} UNION {?x <" + rUri + "> <" + eUri + ">}.}";
        return getQueryStringResult(QueryFactory.create(query, Syntax.syntaxARQ));
    }

    /**
     * get the relation between two entities
     *
     * @param eURI1 entity1
     * @param eURI2 entity2
     * @return the relations between entity1 and entity2
     */
    public static List<String> getRelBetweenEPair(String eURI1, String eURI2) {
        String query = "SELECT ?p WHERE{ { <" + eURI1 + "> ?p <" + eURI2 + "> } UNION {<" + eURI2 + "> ?p <" + eURI1 + ">}.}";
        List<String> queryStringResult = getQueryStringResult(QueryFactory.create(query, Syntax.syntaxARQ));
        queryStringResult.removeIf(uri -> uri.contains("wiki") || Detector.getPredicateBlackList().contains(uri));
        return queryStringResult;

    }

    /**
     * get the label by mid in freebase
     *
     * @param mid mid in freebase
     * @return label in freebase
     */
    public static Set<String> getFreeBaseLabels(String mid) {


        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (!mid.contains("m.")) {
            Set<String> result = new HashSet<>();
            result.add(mid);
            return result;
        }

        if (mid.startsWith("http://rdf.freebase.com/ns/")) {
            mid = mid.replace("http://rdf.freebase.com/ns/", "ns:");
        } else if (mid.startsWith("m.")) {
            mid = "ns:" + mid;
        }

        String sparql = "SELECT ?x where {{" + mid + " rdfs:label ?x}}";
        //+ "UNION {" + mid + " ns:type.object.name ?x} }";
        //+ "UNION {" + mid + " ns:common.topic.alias ?x}}";
        sparql = addPrefix(sparql);
        //System.out.println(sparql);
        List<String> labelList = getQueryStringResult(QueryFactory.create(sparql));

        Set<String> resultSet = new HashSet<>();
        //search for english label
        labelList.forEach(label -> {
            if (label.contains("@en")) {
                resultSet.add(label.split("@")[0]);
            }
        });

        return resultSet;

    }

    public static String getFreeBasePropertyLabel(String property) {


        if (property == null || property.trim().equals("")) {
            return "";
        }

        if (property.startsWith("http://rdf.freebase.com/ns/")) {
            property = property.replace("http://rdf.freebase.com/ns/", "ns:");
        } else if (!property.startsWith("ns:")) {
            property = "ns:" + property;
        }
        String sparql = "SELECT ?x where {{" + property + " rdfs:label ?x}}";
        sparql = addPrefix(sparql);

        //System.out.println(sparql);

        List<String> labelList = getQueryStringResult(QueryFactory.create(sparql));
        Set<String> resultSet = new HashSet<>();

        //search for english label
        for (String label : labelList) {
            if (label.contains("@en")) {
                return label.split("@")[0];

            }
        }

        String[] split = property.split("\\.");
        return split[split.length - 1];

    }

    public static String getMidBykey(String freebase_key) {

        if (freebase_key.startsWith("m.")) { //already mid
            return "http://rdf.freebase.com/ns/" + freebase_key;
        }

        if (freebase_key.startsWith("en.")) { // freebase key
            freebase_key = freebase_key.replace("en.", "/en/");
        }

        String sparql = "select ?x where{ ?x " + "ns:type.object.key \"" + freebase_key + "\"}";
        sparql = addPrefix(sparql);

        List<String> result = getQueryStringResult(QueryFactory.create(sparql));

        if (!result.isEmpty()) {
            return result.get(0);
        } else {
            return null;
        }
    }

    public static Set<String> getOneHopPropertyByMid(String mid) {

        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (mid.startsWith("http://rdf.freebase.com/ns/")) {
            mid = mid.replace("http://rdf.freebase.com/ns/", "ns:");
        } else if (mid.startsWith("m.")) {
            mid = "ns:" + mid;
        }

        String sparql = "SELECT ?p where {{" + mid + " ?p ?x} " +
                "UNION {?y ?p " + mid + "}}";
        sparql = addPrefix(sparql);


        List<String> labelList = getQueryStringResult(QueryFactory.create(sparql));
        labelList.removeIf(uri -> uri.contains("wikipedia"));
        labelList.removeIf(uri -> uri.contains("type.object."));
        labelList.removeIf(uri -> uri.contains("common.topic."));

        return new HashSet<>(labelList);

    }

    public static Set<String> getOutProperty(String mid) {
        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (mid.startsWith("http://rdf.freebase.com/ns/")) {
            mid = mid.replace("http://rdf.freebase.com/ns/", "ns:");
        } else if (mid.startsWith("m.")) {
            mid = "ns:" + mid;
        }

        String sparql = "SELECT ?p where {" + mid + " ?p ?x} ";
        sparql = addPrefix(sparql);


        List<String> propertyList = getQueryStringResult(QueryFactory.create(sparql));
        Set<String> result = new HashSet<>(propertyList);
        filterProperty_Freebase(result);

        return result;
    }

    public static void filterProperty_Freebase(Set<String> propertySet) {
        propertySet.removeIf(uri -> !uri.startsWith("http://rdf.freebase.com/ns/"));
        propertySet.removeIf(uri -> uri.contains("wikipedia"));
        propertySet.removeIf(uri -> uri.contains("type.object."));
        propertySet.removeIf(uri -> uri.contains("common.topic."));
        propertySet.removeIf(uri -> uri.contains("#type"));
        propertySet.removeIf(uri -> uri.contains("#label"));
        propertySet.removeIf(uri -> uri.contains("_id"));
        propertySet.removeIf(uri -> uri.contains("/ns/freebase."));

    }

    public static Set<String> getSecondHopProperty(String mid, String firstHop) {

        String sparql = "select ?p where{ <" + mid + "> <" + firstHop + "> ?c . \n" +
                "?c ?p ?x .\n" +
                "filter(?x != <" + mid + ">)}";

        Set<String> secondHop = new HashSet<>(getQueryStringResult(QueryFactory.create(sparql)));
        filterProperty_Freebase(secondHop);
        return secondHop;

    }

    public static Set<String> getInProperty(String mid) {
        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (mid.startsWith("http://rdf.freebase.com/ns/")) {
            mid = mid.replace("http://rdf.freebase.com/ns/", "ns:");
        } else if (mid.startsWith("m.")) {
            mid = "ns:" + mid;
        }

        String sparql = "SELECT ?p where {?x ?p " + mid + "} ";
        sparql = addPrefix(sparql);

        List<String> labelList = getQueryStringResult(QueryFactory.create(sparql));
        labelList.removeIf(uri -> uri.contains("wikipedia"));
        labelList.removeIf(uri -> uri.contains("type.object."));
        labelList.removeIf(uri -> uri.contains("common.topic."));


        return new HashSet<>(labelList);
    }

    public static Set<String> getOneHopResource(String mid, String property) {
        String sparql = "select ?x where {<" + mid + "> <" + property + "> ?x }";
        return new HashSet<>(getQueryStringResult(QueryFactory.create(sparql)));

    }

    /**
     * judge whether a mid is a CVT node
     *
     * @param mid the mid
     * @return true for CVT; false o.w.
     */
    public static boolean judgeIfCVT(String mid) {

        // CVT must be an mid
        if (!mid.contains("m.")) {
            return false;
        }

        if (mid.startsWith("m.")) {
            mid = "http://rdf.freebase.com/ns/" + mid;
        }

        // judge whether it contains rdf:label
        String sparql = "select ?x where {<" + mid + "> rdfs:label ?x }";
        sparql = addPrefix(sparql);
        return getQueryStringResult(QueryFactory.create(sparql)).isEmpty();

    }

    public static Set<String> getOneOrTwoHopProperty_DBpedia(String entityURL) {

        Set<String> result = new HashSet<>();
        String sparql = "PREFIX ns:<http://rdf.freebase.com/ns/>\n" +
                "select distinct ?p1 ?p2 where\n" +
                "{\n" +
                "\n" +
                "{<" + entityURL + "> ?p1 ?x} UNION {?x ?p1 <" + entityURL + ">}" +
                "UNION\n" +
                "{<" + entityURL + "> ?p1 ?y.\n" +
                " ?y ?p2 ?z.}\n" +
                "\n" +
                "filter(?z != <" + entityURL + ">)\n" +
                "filter(?p1 != rdf:type && ?p1 != rdfs:label)\n" +
                "filter(?p2 != rdf:type && ?p2 != rdfs:label)\n" +
                "filter(?p1 != ns:type.object.type && ?p1 != ns:type.object.instance)\n" +
                "filter(?p2 != ns:type.object.type && ?p2 != ns:type.object.instance)\n" +
                "filter( !regex(?p1,\"wikipedia\",\"i\"))\n" +
                "filter( !regex(?p2,\"wikipedia\",\"i\"))\n" +
                "filter( !regex(?p1,\"type.object\",\"i\"))\n" +
                "filter( !regex(?p2,\"type.object\",\"i\"))\n" +
                "filter( !regex(?p1,\"common.topic.\",\"i\"))\n" +
                "filter( !regex(?p2,\"common.topic.\",\"i\"))\n" +
                "filter( !regex(?p1,\"_id\",\"i\"))\n" +
                "filter( !regex(?p2,\"_id\",\"i\"))\n" +
                "filter( !regex(?p1,\"#type\",\"i\"))\n" +
                "filter( !regex(?p2,\"#type\",\"i\"))\n" +
                "filter( !regex(?p1,\"#label\",\"i\"))\n" +
                "filter( !regex(?p2,\"#label\",\"i\"))\n" +
                "filter( !regex(?p1,\"_id\",\"i\"))\n" +
                "filter( !regex(?p2,\"_id\",\"i\"))\n" +
                "filter( !regex(?p1,\"/ns/freebase.\",\"i\"))\n" +
                "filter( !regex(?p2,\"/ns/freebase.\",\"i\"))\n" +
                "\n" +
                "\n" +
                "filter( regex(?p1,\"^http://rdf.freebase.com/ns/\",\"i\"))\n" +
                "\n" +
                "\n" +
                "filter ( not exists {?y ns:type.object.name ?m} )\n" +
                "filter ( exists {?x ns:type.object.name ?m} || not exists {?x ?n ?m} )\n" +
                "\n" +
                "}";
        sparql = addPrefix(sparql);
        return result;
    }

    public static Set<String> getOneOrTwoHopPropertyByMid(String mid) {

        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (mid.startsWith("m.")) {
            mid = "http://rdf.freebase.com/ns/" + mid;
        }


        Set<String> result = new HashSet<>();

        Set<String> oneHopPropertyByMid = getOutProperty(mid);
        //System.out.println("oneHopSize:" + oneHopPropertyByMid.size());
        //System.out.println("oneHopProperties:" + oneHopPropertyByMid);

        for (String property : oneHopPropertyByMid) {
            //System.out.println(property);
            Set<String> oneHopEntity = getOneHopResource(mid, property);
            if (!oneHopEntity.isEmpty()) {
                boolean allCVT = true; // whether all the resources are CVT
                for (String resource : oneHopEntity) {
                    if (!judgeIfCVT(resource)) {// CVT node
                        allCVT = false;
                        break;
                    }
                }

                //not CVT
                if (!allCVT) {
                    result.add(property);
                } else {// CVT, two hop
                    Set<String> secondHopProperty = getSecondHopProperty(mid, property);
                    secondHopProperty.forEach(secondProp ->
                            result.add(property + "||" + secondProp));
                }
            }
        }

        return result;
    }

    public static Set<String> getOneOrTwoHopPropertyByMid_New(String mid) {

        if (mid == null || mid.trim().equals("")) {
            return new HashSet<>();
        }

        if (mid.startsWith("m.")) {
            mid = "http://rdf.freebase.com/ns/" + mid;
        }

        Set<String> result = new HashSet<>();
        String sparql = "PREFIX ns:<http://rdf.freebase.com/ns/>\n" +
                "select distinct ?p1 ?p2 where\n" +
                "{\n" +
                "\n" +
                "{<" + mid + "> ?p1 ?x}\n" +
                "UNION\n" +
                "{<" + mid + "> ?p1 ?y.\n" +
                " ?y ?p2 ?z.}\n" +
                "\n" +
                "filter(?z != <" + mid + ">)\n" +
                "filter(?p1 != rdf:type && ?p1 != rdfs:label)\n" +
                "filter(?p2 != rdf:type && ?p2 != rdfs:label)\n" +
                "filter(?p1 != ns:type.object.type && ?p1 != ns:type.object.instance)\n" +
                "filter(?p2 != ns:type.object.type && ?p2 != ns:type.object.instance)\n" +
                "filter( !regex(?p1,\"wikipedia\",\"i\"))\n" +
                "filter( !regex(?p2,\"wikipedia\",\"i\"))\n" +
                "filter( !regex(?p1,\"type.object\",\"i\"))\n" +
                "filter( !regex(?p2,\"type.object\",\"i\"))\n" +
                "filter( !regex(?p1,\"common.topic.\",\"i\"))\n" +
                "filter( !regex(?p2,\"common.topic.\",\"i\"))\n" +
                "filter( !regex(?p1,\"_id\",\"i\"))\n" +
                "filter( !regex(?p2,\"_id\",\"i\"))\n" +
                "filter( !regex(?p1,\"#type\",\"i\"))\n" +
                "filter( !regex(?p2,\"#type\",\"i\"))\n" +
                "filter( !regex(?p1,\"#label\",\"i\"))\n" +
                "filter( !regex(?p2,\"#label\",\"i\"))\n" +
                "filter( !regex(?p1,\"_id\",\"i\"))\n" +
                "filter( !regex(?p2,\"_id\",\"i\"))\n" +
                "filter( !regex(?p1,\"/ns/freebase.\",\"i\"))\n" +
                "filter( !regex(?p2,\"/ns/freebase.\",\"i\"))\n" +
                "\n" +
                "\n" +
                "filter( regex(?p1,\"^http://rdf.freebase.com/ns/\",\"i\"))\n" +
                "\n" +
                "\n" +
                "filter ( not exists {?y ns:type.object.name ?m} )\n" +
                "filter ( exists {?x ns:type.object.name ?m} || not exists {?x ?n ?m} )\n" +
                "\n" +
                "}";
        sparql = addPrefix(sparql);
        //System.out.println(sparql);

        Query query = QueryFactory.create(sparql);

        if (connection == null) {
            connection = new VirtGraph(CORE_KB_GRAPH, CORE_KB_JDBC, "dba", "dba");
        }
        VirtuosoQueryExecution vqe = VirtuosoQueryExecutionFactory.create(query, connection);
        ResultSet resultSet = vqe.execSelect();
        while (resultSet.hasNext()) {
            QuerySolution querySolution = resultSet.nextSolution();

            String property = null;
            RDFNode p1 = querySolution.get("p1");
            RDFNode p2 = querySolution.get("p2");
            if (p2 == null) {
                property = p1.toString();
            } else {
                property = p1.toString() + "||" + p2.toString();
            }
            result.add(property);
        }
        filterProperty_Freebase(result);
        return result;
    }
}



