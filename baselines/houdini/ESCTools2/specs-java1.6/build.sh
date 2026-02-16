#!/bin/bash
jar cvf ../../mobius.escjava2.esctools/escspecs-java1.6.jar `find ./ -name *.jml` `find ./ -name *.refines-java`  `find ./ -name *.spec` `find ./ -name *.refines-spec`
