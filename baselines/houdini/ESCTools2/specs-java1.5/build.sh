#!/bin/bash
jar cvf ../escspecs-java1.5.jar `find ./ -name *.jml` `find ./ -name *.refines-java`  `find ./ -name *.spec` `find ./ -name *.refines-spec`
