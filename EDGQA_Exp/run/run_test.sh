#!/bin/bash
cd ../EDGQA
mvn package -o  

screen_test="EDGQA_Test"
screen -dmS $screen_test

 cmd=" java -jar  target/EDGQA-1.0-SNAPSHOT.jar -d 'lc-quad'\
 -tr 'false' -r 'autotest' -cc 'false' -uc 'true' -gll 'true' -lll 'true' -qd 'true' -rr 'true'\
 -file '../QDT2EDG_convert/converted/clue_decipher_converted.json'"
screen -x -S $screen_test -p 0 -X stuff "$cmd"
screen -x -S $screen_test -p 0 -X stuff $'\n'

 cmd="java -jar  target/EDGQA-1.0-SNAPSHOT.jar -d 'lc-quad'\
 -tr 'false' -r 'autotest' -cc 'false' -uc 'true' -gll 'true' -lll 'true' -qd 'true' -rr 'true'\
 -file '../QDT2EDG_convert/converted/decomprc_converted.json'"

screen -x -S $screen_test -p 0 -X stuff "$cmd"
screen -x -S $screen_test -p 0 -X stuff $'\n'

cmd="java -jar  target/EDGQA-1.0-SNAPSHOT.jar -d 'lc-quad'\
 -tr 'false' -r 'autotest' -cc 'false' -uc 'true' -gll 'true' -lll 'true' -qd 'true' -rr 'true'\
 -file '../QDT2EDG_convert/converted/HSP_converted.json'"
screen -x -S $screen_test -p 0 -X stuff "$cmd"
screen -x -S $screen_test -p 0 -X stuff $'\n'

cmd=" java -jar  target/EDGQA-1.0-SNAPSHOT.jar -d 'lc-quad'\
 -tr 'false' -r 'autotest' -cc 'false' -uc 'true' -gll 'true' -lll 'true' -qd 'true' -rr 'true'\
 -file '../QDT2EDG_convert/converted/splitqa_converted.json'"
screen -x -S $screen_test -p 0 -X stuff "$cmd"
screen -x -S $screen_test -p 0 -X stuff $'\n'

