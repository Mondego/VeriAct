#! /usr/bin/env python                                                          
import sys
import os

fn = sys.argv[1]

os.system("mv "+fn+" "+fn+"_old")

f1leSource = open(fn+"_old",'r+')

f1leDest = open(fn,'w')

debutNewFile = ""
endNewFile = ""

inEndOfFile = False
wantThisLine = True

for line in f1leSource :

    wantThisLine = True

    if line.find("Start of pvs declarations") != -1 :
        inEndOfFile = True
        wantThisLine = False

    if wantThisLine :
        if inEndOfFile :
            endNewFile = endNewFile +line
        else :
            debutNewFile = debutNewFile + line

f1leDest.write(endNewFile)
f1leDest.write(debutNewFile)

f1leDest.close()
f1leSource.close()

os.system("rm "+fn+"_old -f")

