
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

class KBExecutor():
    def __init__(self):
        pass
    def normalize_ent_link(self,link):
        pass
    def query_db(self,sparql_query):
        pass
    def query_relation_degree(self,relation):
        pass
    def query_onehop_relation(self,link):
        pass
    def get_link_label(self,link, entity2label=None):
        pass
    def answer_convert(self,item_answer):
        if 'boolean' in item_answer.keys():
            var = 'boolean'
            return item_answer['boolean']
        else:
            answer = []
            var_list = item_answer['head']['vars']
            if len(var_list) == 1:
                var = var_list[0]
                if var == 'boolean':
                    answer.append(item_answer['boolean'])
                else:
                    for cand in item_answer['results']['bindings']:
                        if var == 'date':
                            answer.append(cand['date']['value'])
                        elif var == 'number':
                            answer.append(cand['c']['value'])
                        elif var == \
                                'resource' or var == 'uri':
                            answer.append(cand['uri']['value'])
                        elif var == 'string':
                            answer.append(cand['string']['value'])
                        elif var == 'callret-0':
                            answer.append(cand['callret-0']['value'])
                        else:
                            answer.append(cand[var]['value'])
            else:
                for cand in item_answer['results']['bindings']:
                    dic_temp = {}
                    for var in var_list:
                        dic_temp[var] = cand[var]['value']
                    answer.append(dic_temp)
        return answer

class FBExecutor(KBExecutor):
    def __init__(self):
        KBExecutor.__init__(self)
        self.endpoint='http://210.28.134.34:8890/sparql'
    def normalize_ent_link(self,link):
        KBExecutor.normalize_ent_link(self,link)
        if 'http://rdf.freebase.com/' in link or 'www.freebase.com/' in link:
            link = link[link.rfind('/') + 1:]
        elif link[:4] == 'ns:m' or link[:4] == 'ns:g':
            link = link[3:]
        else:
            link = link
        return link
    def query_db(self,sparql_query):
        KBExecutor.query_db(self,sparql_query)

        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()  # json,type为dict
        answer = self.answer_convert(results)  # ;df=pd.DataFrame.from_dict(answer)
        return answer
    def query_relation_degree(self,relation):
        KBExecutor.query_relation_degree(self,relation)

        sparql_query="PREFIX ns: <http://rdf.freebase.com/ns/> select count(?s) where{?s "+relation+" ?o}"
        answer=int(self.query_db(sparql_query)[0])
        return answer
    def query_onehop_relation(self,link):
        KBExecutor.query_onehop_relation(self,link)
        link=self.normalize_ent_link(link)

        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery("""PREFIX ns: <http://rdf.freebase.com/ns/>
                            SELECT DISTINCT  ?p ?o
                            WHERE {FILTER (!isLiteral(?o) OR lang(?o) = '' OR langMatches(lang(?o), 'en'))
                            ns:"""+link+""" ?p ?o .
                            }""")
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()  # json,type为dict
        answer = self.answer_convert(results)
        df=pd.DataFrame.from_dict(answer)
        pvc=df.p.value_counts()
        return pvc

    def get_link_label(self,link, entity2label=None):
        KBExecutor.get_link_label(self,link)

        link=self.normalize_ent_link(link)

        if entity2label != None:
            answer = entity2label.get(link)
            if answer != None:
                return answer
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(
            """PREFIX ns: <http://rdf.freebase.com/ns/>  SELECT DISTINCT ?x  WHERE {FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')). ns:""" + str(
                link) + """     rdfs:label ?x .}""")
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()  # json,type为dict
        answer = self.answer_convert(results)
        # df = pd.DataFrame.from_dict(answer)
        return answer



