function NDONE () {
    ls -d $LIBRERESULTSDIR/*/ | while read d; do echo $( echo "$d" ; ls $d | wc -l ; ls -ltr $d | tail -n 1 | awk '{print $6 $7}' ); done
}

function NDONEM () {
    ls -d $LIBREMETRICSDIR/*/ | while read d; do echo $( echo "$d" ; ls $d | wc -l ; ls -ltr $d | tail -n 1 | awk '{print $6 $7}' ); done
}

