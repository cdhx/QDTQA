
import json
from strsimpy import NormalizedLevenshtein
from strsimpy.jaro_winkler import JaroWinkler


#some tools for lexical similarity
def dice_coefficient(a, b):
    """dice coefficient 2nt/(na + nb)."""
    if not len(a) or not len(b): return 0.0
    if len(a) == 1:  a = a + u'.'
    if len(b) == 1:  b = b + u'.'

    a_bigram_list = []
    for i in range(len(a) - 1):
        a_bigram_list.append(a[i:i + 2])
    b_bigram_list = []
    for i in range(len(b) - 1):
        b_bigram_list.append(b[i:i + 2])

    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return dice_coeff


def jaro_winkler_similarity(text_a: str, text_b: str):
    text_a = text_a.lower()
    text_b = text_b.lower()
    return jaro_winkler.similarity(text_a, text_b)


def normalized_levenshtein_similarity(text_a: str, text_b: str):
    return normalized_levenshtein.similarity(text_a, text_b)


def literal_similarity(text_a: str, text_b: str, substr_match=False, log=None):
    text_a = text_a.replace('<e>', '').replace('  ', '')
    text_b = text_b.replace('<e>', '').replace('  ', '')

    if substr_match is True:
        if text_a in text_b or text_b in text_a:
            return 1
    jaro_sim = jaro_winkler.similarity(text_a, text_b)
    norm_lev = normalized_levenshtein.similarity(text_a, text_b)
    dice_sim = dice_coefficient(text_a, text_b)
    res = (jaro_sim + norm_lev + dice_sim) / 3
    assert 0 <= res <= 1
    return res


jaro_winkler = JaroWinkler()
normalized_levenshtein = NormalizedLevenshtein()


# ---------EDG FORMAT-------------------
# {
#     "taggedQuestion": "Which globe region with the UTCâ\u0088\u009205:00 time zone does the Dominican Republic call home?",
#     "entityMap": {"<e0>": "UTC 05:00 time zone"},
#     "nodeNum": 3,
#     "nodes": [
#         {
#             "containsRefer": false,
#             "start": -1,
        
#             "entityID": -1,
#             "end": -1,
#             "trigger": "which",
#             "nodeType": 1,
#             "nodeID": 0,
#             "questionType": "COMMON"
#         },
#     ],
#     "question": "Which globe region with the UTCâ\u0088\u009205:00 time zone does the Dominican Republic call home?",
#     "entityNum": 1,
#     "edges": [
#         {
#             "edgeType"
#             "isEqual"
#             "start"
#             "from"
#             "end"
#             "to"
#         },
#     ],
#     "sparql_query": 
#     "syntaxTreeText":
#     "id": 17
# },

class EDGnode():    # structure of an EDG-format node
    def __init__(self):
        self.str = self.originStr = None
        self.refer = False
        self.start = self.end = -1
        self.nodeType = self.nodeID = self.entityID = -1
        self.questionType = "UNKNOWN"
        self.trigger = None

    def toDict(self):   # convert a EDG-format node to a dict
        D = {"containsRefer": self.refer,"start": self.start, "entityID": self.entityID,"end": self.end,"nodeType": self.nodeType, "nodeID": self.nodeID, "questionType": self.questionType}
        if self.trigger is not None:
            D['trigger']=self.trigger
        if self.str is not None:
            D['str'] = self.str
        if self.originStr is not None:
            D['originStr'] = self.originStr
        return D

    def loadFromEDGDict(self, EDGdict): # load an EDG-format node from a dict
        if 'str' in EDGdict:
            self.str = EDGdict['str']
        if 'originStr' in EDGdict:
            self.originStr = EDGdict['originStr']
        if 'containsRefer' in EDGdict:
            self.refer = EDGdict[ 'containsRefer']
        self.start = EDGdict['start']
        self.end = EDGdict['end']
        self.nodeType = EDGdict['nodeType']
        self.nodeID = EDGdict['nodeID']
        self.questionType = EDGdict['questionType']
        if 'trigger' in EDGdict:
            self.trigger = EDGdict['trigger']


class EDGedge():    #structure of an EDG-format edge.
    def __init__(self):
        self.edgeType = self.start=self.end = self.fr = self.to = -1
        self.isEqual = False
    def toDict(self):   # convert to dict
        D = {"edgeType": self.edgeType, "isEqual": self.isEqual, "start": self.start, "from": self.fr, "end": self.end, "to": self.to}
        return D


#some special decomposition marks, punctuation marks and meaningless words. (For additional post-processing)
STRUCTURES = ['[INQR]', '[INQL]', '[DES]']
SYMBOLS = ['.', ',' ,'?']
WH_LOWER = ['what', "which", "who", "whom", 'whose']
PREPS = ['of', 'in', 'on', 'that', 'and' , 'or', 'but', 'also']
VERB = ['are', 'is' ,'were', 'was', 'do', 'did', 'does', 'have',  'has', 'had']


# Processing a desc
def process_desc(desc, delete_structures = False, delete_symbols = False, delete_grammar_element = False):
    desc = desc.replace('[ENT]',"#entity1")
    if delete_structures:
        for struct in STRUCTURES:
            desc = desc.replace(struct, "")
    if delete_symbols:
        for symbol in SYMBOLS:
            if desc.endswith(symbol):
                desc = desc[0:-1]
                break
    if delete_grammar_element:
        #remove meaningless grammar elements, especially those on the beginning of the desc
        tokens = desc.split()
        if len(tokens) == 0:
            return desc
        begin = tokens[0]
        second = None
        if len(tokens) > 1:
            second = tokens[1]
        # what is ...
        if second is not None and begin.lower() in WH_LOWER and second.lower() in VERB:
            desc = desc.replace(begin, "", 1)
            desc = desc.replace(second, "", 1)
            return desc
        # of which / of whom..
        if second is not None and begin.lower() in PREPS and second.lower() in WH_LOWER:
            desc = desc.replace(begin, "", 1)
            desc = desc.replace(second, "", 1)
            return desc
        # what .. / in of .../ is
        if begin.lower() in WH_LOWER or begin.lower() in PREPS or begin.lower() in VERB:
            desc = desc.replace(begin, "", 1)
            return desc
    return desc

#Convert decomposed files from EDG format to QDT format.
#arguments:
#       QDT file: QDT-format decomposition file to be converted
#
#       output_file：The path of output EDG-format file
#
#       match_by_index：Find the EDG-format decomposition corresponding to the QDT-format decomposition of a problem in the order it is listed in the json file.
#                       If not, the search will use question tokens.
#
#       edg_example:Decomposition of EDGQA on LC-Quad test set.
#                  EDGQA may require additional information such as trigger words, which are only relevant to the problem before decomposition.
#                  So here we use EDGQA's own decomposition results to import trigger words, etc.
#
#       composition_file:Type judgments of the bert classifier for each question on the LC-Quad test set. If the classifier discriminates the problem as a simple type,
#                       a 'simple' style decomposition is used instead of the original model's decomposition of the problem.
#
#       delete_elements: Remove some opening words similarly to EDGQA decomposition (not used in the experiment)

def convert_QDT_to_EDG(QDTfile, output_file, match_by_index=True, edg_example = None, composition_file = None, delete_elements = False):

    with open(edg_example) as f1:
        EDGdata = json.load(f1)

    with open(QDTfile) as f2:
        QDTdata = json.load(f2)

    if composition_file is not None:
        with open(composition_file) as f3:
            composition_file = json.load(f3)
        simple_as_one = True        # 'simple' questions will be decomposed into "root -> entity ->desc"
    else:
        composition_file = []
        simple_as_one = False       # follow original decompostion result 
    datas = []
    EDG_dict = {}
    compositionality_dict = {}
    if not match_by_index:
        for d in EDGdata:
            question_lower = d['question'].lower()
            for k in range(0, 3):
                question_lower =  question_lower.strip("?")
                question_lower =  question_lower.strip(".")
                question_lower = question_lower.strip()
            EDG_dict[question_lower] = d
        if simple_as_one:
            for d in composition_file:
                question_lower = d['question'].lower()
                for k in range(0, 3):
                    question_lower =  question_lower.strip("?")
                    question_lower =  question_lower.strip(".")
                    question_lower = question_lower.strip()
                compositionality_dict[question_lower] = d
    print(len(composition_file))
    print(QDTfile)
    print(len(QDTdata))
    print(len(EDGdata))
    for i in range(0, len(QDTdata)):
        id = i
        edgExample = None
        if match_by_index is True:
            if simple_as_one is True:
                simple_check = composition_file[i]
            EDG_example = EDGdata[i]
        else:
            #Use the question tokens to find the corresponding EDG-format decompostion
            question_lower = QDTdata[i]['question'].lower()
            for k in range(0, 3):
                question_lower =  question_lower.strip("?")
                question_lower =  question_lower.strip(".")
                question_lower = question_lower.strip()
            if question_lower in EDG_dict:      #find correctly
                print("EDG:find",question_lower)
                EDG_example = EDG_dict[question_lower]
            else:
                print("EDG:cannot find",question_lower,"use bruteforce")
                q1 = QDTdata[i]['question']     #else, we adopt violent search using literal similarity
                max_sim = -1
                idx = -1
                for j in  range(0, len(EDGdata)):
                    q2 = EDGdata[j]['question']
                    sim = literal_similarity(q1,q2,substr_match=True)
                    if sim > max_sim:
                        max_sim = sim
                        idx = j
                print("most likely in", idx, " ",EDGdata[idx]['question'],"with similarity ",max_sim)
                EDG_example = EDGdata[idx]
            id = EDG_example['id']
            if simple_as_one is True:
                if question_lower in compositionality_dict:
                    print("QDT:find",question_lower)
                    simple_check = compositionality_dict[question_lower]
                else:
                    simple_check = composition_file[idx]
        question = QDTdata[i]['question']
        tagged_question = EDG_example['taggedQuestion']
        entity_map = EDG_example['entityMap']
        if 'syntaxTreeText' in EDG_example:
            syntax_tree = EDG_example['syntaxTreeText']
        nodenum=len(QDTdata[i]['edg']['nodes'])
        edgenum=len(QDTdata[i]['edg']['edge'])
        sparql_query = ""
        if 'sparql' in QDTdata[i]:
            sparql_query = QDTdata[i]['sparql']
        EDGnodes = []
        EDGedges = []

    #deal with nodes
        # need to add an extra root for QDT->EDG convert
        root = EDGnode()
        for n in EDG_example['nodes']:
            if n['nodeID'] == 0:
                root.loadFromEDGDict(n)
                break
        EDGnodes.append(root)
        if simple_as_one is True and simple_check['compositionality_type'] == 'simple':  #gen simple decomposition  root -> ent -> desc
            entity_node = EDGnode()
            entity_node.nodeID = 1
            entity_node.entityID = 0
            entity_node.nodeType = 2 #ent
            desc_node = EDGnode()
            desc_node.entityID = 0
            desc_node.nodeID = 2
            desc_node.nodeType = 3 #desc
            desc_node.refer = False
            desc_node.originStr = desc_node.str = process_desc(question, delete_structures=True, delete_symbols=True, delete_grammar_element=delete_elements)
            e1 = EDGedge()
            e1.fr = 0
            e1.to = 1
            e1.edgeType = 1
            e2 = EDGedge()
            e2.fr = 1
            e2.to = 2
            e2.edgeType = 3
            EDGnodes.append(entity_node)
            EDGnodes.append(desc_node)
            EDGedges.append(e1)
            EDGedges.append(e2)
        else:                           # generate EDG-format according to corresponding QDT-format 
            for n in QDTdata[i]['edg']['nodes']:
                #deal with nodes
                node = EDGnode()
                node.nodeID = n['nodeID']+1 # ID += 1 because of the exist of new root node
                node.entityID = n['entityID']
                if 'value' in n:
                    node.refer = n['hasRefer']
                    node.originStr = node.str = process_desc(n['value'], delete_structures=True, delete_symbols=True, delete_grammar_element=delete_elements)
                #deal with nodeType
                if n['nodeType'] == "Entity":   # entity -> entity
                    node.nodeType = 2
                else:                           # desc -> verb desc / non-verb desc 
                    if node.str is None or node.str is "":
                        print('[ERROR] empty description node, type:',n['nodeType'])
                    verb_desc = False
                    if verb_desc:
                        node.nodeType = 3
                    else:
                        node.nodeType = 4
                EDGnodes.append(node)
            #deal with edges
            inner_entityID = []
            for e in QDTdata[i]['edg']['edge']:
                edge = EDGedge()
                edge.fr = e['from'] + 1 #  because of the exist of new ROOT node
                edge.to = e['to'] + 1
                from_type = None
                to_type = None
                #deal with edgeTypes
                for node in EDGnodes:
                    if node.nodeID == edge.fr:
                        from_type = node.nodeType
                    if node.nodeID == edge.to:
                        to_type =  node.nodeType
                        if node.nodeType == 2:
                            inner_entityID.append(node.nodeID)
                if from_type == 1:
                    edge.edgeType = 1
                else:
                    edge.edgeType = to_type
                EDGedges.append(edge)
            #add extra edges from root to entity
            for node in EDGnodes:
                if node.nodeType == 2 and node.nodeID not in inner_entityID:
                    e = EDGedge()
                    e.fr = 0
                    e.to = node.nodeID
                    e.edgeType = 1
                    EDGedges.append(e)

        #assemble these datas into a json object
        json_object = {}
        node_num = len(EDGnodes)
        edge_num = len(EDGedges)
        entity_num = -1
        node_dict_list = []
        edge_dict_list = []
        for node in EDGnodes:
            entity_num = max(entity_num, node.entityID+1)
            node_dict_list.append(node.toDict())
        for edge in EDGedges:
            edge_dict_list.append(edge.toDict())
        json_object = {}
        json_object['taggedQuestion'] = tagged_question
        json_object['entityMap'] = entity_map
        json_object['nodeNum'] = node_num
        json_object['nodes']=node_dict_list
        json_object['question'] = question
        json_object['edges'] = edge_dict_list
        json_object['sparql_query'] = sparql_query
        json_object['syntaxTreeText'] = syntax_tree
        json_object['entityNum'] = entity_num
        json_object['id'] = id
        datas.append(json_object)

    with open(output_file,'w') as f:    #write into file
        datas.sort(key=lambda item: item['id'])
        json.dump(datas, f, indent=1)
    return 


#If the type of a question in (simple_QDT_file) is simple, replace the decomposition of the question in (QDT_file) with the simple type
def convert_QDT_to_simple_QDT(QDTfile, simple_QDT_file, output):
    with open(QDTfile) as f1:
        data1 = json.load(f1)
    with open(simple_QDT_file) as f2:
        data2 = json.load(f2)
    
    for i in range(0, len(data1)):
        if data2[i]['compositionality_type'] == 'simple':
            data1[i] = data2[i]
    with open(output,'w') as f:
        json.dump(data1, f, indent=1)
    return 

#usage examples

# Since the comparison method is difficult to determine whether a question needs to be decomposed or not, 
# we trained a bert classifier to determine question types. If the predicted type of a question is "simple",
#  the question decomposition in the comparison method will be replaced with the "simple" type decomposition to improve the performance of the comparison method.
composition_file = "resources/composition/lc_pred_type_test.json"

qdt = "resources/QDT_format/clue_decipher.json"
convert_QDT_to_EDG(QDTfile=qdt, output_file="converted/clue_decipher_converted1.json", edg_example = "resources/EDG/EDG_lc_test_decom.json",
            composition_file=None, delete_elements=False)

qdt = "resources/QDT_format/decomprc.json"
convert_QDT_to_EDG(QDTfile=qdt, output_file="converted/decomprc_converted.json", edg_example = "resources/EDG/EDG_lc_test_decom.json",
            composition_file=composition_file, delete_elements=False)

qdt = "resources/QDT_format/HSP.json"
convert_QDT_to_EDG(QDTfile=qdt, output_file="converted/HSP_converted.json", edg_example = "resources/EDG/EDG_lc_test_decom.json",
            composition_file=composition_file, delete_elements=False)

qdt = "resources/QDT_format/splitqa.json"
convert_QDT_to_EDG(QDTfile=qdt, output_file="converted/splitqa_converted.json",match_by_index=False, edg_example = "resources/EDG/EDG_lc_test_decom.json",
            composition_file=composition_file, delete_elements=False)
