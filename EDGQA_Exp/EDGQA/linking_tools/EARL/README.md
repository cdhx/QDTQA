EARL (Entity and Relation Linker), is a system for jointly linking entities and relations in a question to a knowledge graph.
It's source code is available at https://github.com/AskNowQA/EARL .

### 1. Consuming EARL live API in EDGQA

Try this command to check whether the EARL live API is working (might be slow in some areas):
```shell
curl -XPOST 'http://ltdemos.informatik.uni-hamburg.de/earl/processQuery' -H 'Content-Type:application/json' -d'{"nlquery":"Who is the president of Russia?"}'
```

To consume this api in EDGQA, change the `earlURL` to `earlLiveURL` in file `src/main/java/cn/edu/nju/ws/edgqa/utils/linking/LinkingTool.java`.

### 2. Run EARL locally
Please refer to [the official repository of EARL](https://github.com/AskNowQA/EARL).
Follow the instructions and start the linking service on port 4999.

To check whether EARL linking is available, run 
```shell
curl -XPOST 'localhost:4999/processQuery' -H 'Content-Type: application/json' -d '{"nlquery":"the homeground of football team Panionios GSS"}'
```

If the linking results are returned, it means the EARL system is set up correctly.

Finally, remember to set the `earlServerIP` and `earlLocalUrl` in
`src/main/java/cn/edu/nju/ws/edgqa/utils/linking/LinkingTool.java`.
