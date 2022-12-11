Dexter is a framework that implements some popular algorithms and provides all the tools needed to develop any entity linking technique.

It's source code is avaialbe at: https://github.com/dexter/dexter.
It does not offer any live api, so we have to deploy it locally.

### Run Dexter Locally
To start the dexter entity linking service, 
just download the model with the binaries:

```shell
wget http://hpc.isti.cnr.it/~ceccarelli/dexter2.tar.gz
tar -xvzf dexter2.tar.gz
cd dexter2
java -Xmx4000m -jar dexter-2.1.0.jar
```
It will start a service listening to port `8080`.
To check whether the service is available, 
run 
```
curl -XPOST 'http://127.0.0.1:8080/dexter-webapp/api/rest/spot' -H 'Content-Type: application/json' --data-urlencode "text=Brazilian state-run giant oil company Perobras signed a three-year technology and research cooperation agreement with oil service provider Halliburton."   --data "wn=false"   --data "debug=false"
```

For more information, please refer to the official repository of dexter: https://github.com/dexter/dexter

### Setting in EDGQA
Remember to set the `dexterServerIP` and `dexterLocalUrl` in
`src/main/java/cn/edu/nju/ws/edgqa/utils/linking/LinkingTool.java`.


