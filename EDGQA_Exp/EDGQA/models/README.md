#### Content

This directory contains the semantic matching model and block query reranking model.

We apply the official bert classifier to measuring the semantic similarity or correlation between two components (such
as natural language phrases and KB relations).

#### Usage

1. cd to this directory
   ```
   cd models
   ``` 

2. download the checkpoints from [here](https://drive.google.com/file/d/1CV5W19Ghj6i_JHt7sDenVfTjPYql44R-/view?usp=sharing) and unzip them under current directory.


3. run

   ```
   pip install -r requirements
   ```
   to install all the required packages

4. run
   ```
   python relation_semantic_matching_service.py
   ``` 
   to start the relation semantic matching model. This will start a Flask
   service, listening to port 5682

5. run 
   ```
   python block_query_reranker_lcquad_service.py
   ``` 
   to start the block query reranker for lcquad, listening to port
   5683

6. run 
   ```
   python block_query_reranker_qald_service.py
   ``` 
   to start the block query reranker for qald, listening to port 5684

### Test

1. test the relation semantic matching service
   by   
   ```shell
   curl -XPOST '127.0.0.1:5682/relation_detection' -H 'Content-Type: application/json' -d '{"question": "who is the hoster of", "labels": ["hoster","visitor"]}'
   ```
2. test the block query reranking service for lcquad
   by 
   ```shell
   curl -XPOST '127.0.0.1:5683/query_rerank' -H 'Content-Type: application/json' -d '{"edg_block": "[BLK]  [DES] are the bands associated with #entity1 [BLK]  [DES] the artists of My Favorite Girl", "sparql_queries": ["\t [TRP] ?e1 Artist My Favorite Girl (Dave Hollister song) [TRP] ?e0 associated musical artist ?e1"]}'
   ```
3. test the block query reranking service for qald
   by 
   ```shell   
   curl -XPOST '127.0.0.1:5684/query_rerank' -H 'Content-Type: application/json' -d '{"edg_block": "[BLK]  [DES] are the bands associated with #entity1 [BLK]  [DES] the artists of My Favorite Girl", "sparql_queries": ["\t [TRP] ?e1 Artist My Favorite Girl (Dave Hollister song) [TRP] ?e0 associated musical artist ?e1"]}'
   ```

