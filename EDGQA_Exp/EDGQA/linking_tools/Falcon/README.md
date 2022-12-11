FALCON is an entity and relation linking framework over DBpedia.
It's source code is avaialbe at https://github.com/AhmadSakor/falcon.
Also, it offers a live api at https://labs.tib.eu/falcon/.

### 1. Consume Falcon Live API in EDGQA
Try this command to check whether the Falcon live API is working (might be slow in some areas):
```shell
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"text":"Who painted The Storm on the Sea of Galilee?"}' \
  https://labs.tib.eu/falcon/api?mode=long&k=3
```

To consume this api in EDGQA, change the `falconURL` to `falconLiveURL` in file `src/main/java/cn/edu/nju/ws/edgqa/utils/linking/LinkingTool.java`.

### 2. Run Falcon Locally
Please refer to the official repository of Falcon:https://github.com/AhmadSakor/falcon.
Follow the instructions to set up the system.

Since the repository does not contain the service api, we implemented a falcon service in `falcon_service.py`.
Copy it to the falcon directory, and change the dbpedia endpoints in it:
```python
dbpediaSPARQL = "http://210.28.134.34:8892/sparql"
dbpediaSPARQL2 = "http://210.28.134.34:8892/sparql"
```

Then from the falcon directory, run
```shell
python falcon_service.py
```
It will start a service listening to port 9876.
To check whether the service is available, run

```shell
curl -XPOST -H 'Content-Type: application/json' -d '{"text":"In how many places can I find people whose alma mater was in bachelor of arts?","k":3}' http://localhost:9876/annotate
```

If the linking results are returned, it means the falcon system is set up correctly.

Finally, remember to set the `falconServerIP` and `falconLocalUrl` in
`src/main/java/cn/edu/nju/ws/edgqa/utils/linking/LinkingTool.java`.