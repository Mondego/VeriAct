<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE stylesheet [
<!ENTITY nbsp 
"<xsl:text> </xsl:text>">
]>
<!--xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:exslt="http://exslt.org/common" version="1.0"-->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    <!-- WARNING: This stylesheet is work in progress!! -->
    <!-- Author: Carl Pulley -->
    <xsl:output method="xml" omit-xml-declaration="yes" indent="yes"/>
    <!--xsl:output method="text" omit-xml-declaration="yes"/-->
    <xsl:strip-space elements="VCTERM"/>
    <xsl:template match="/">
        <!--xsl:variable name="s-term"-->
          <sterm><xsl:apply-templates/></sterm>
        <!--/xsl:variable-->
        <!--xsl:apply-templates select="exslt:node-set($s-term)" mode="sterm"/-->
    </xsl:template>
    <xsl:template match="DECLN"/>
    <xsl:template match="PROP">
        <list><atom><xsl:value-of select="@name"/></atom> <xsl:apply-templates select="PROP|QUANT|PRED|CONST[@type='Boolean']"/></list>
    </xsl:template>
    <xsl:template match="QUANT[@name='FORALL' or @name='EXISTS']">
        <list><atom><xsl:value-of select="@name"/></atom> <xsl:apply-templates select="VAR"/> <xsl:apply-templates select="PAT"/> <xsl:apply-templates select="BODY"/></list>
    </xsl:template>
    <xsl:template match="PAT">
        <xsl:apply-templates select="PROP|QUANT|PRED|CONST[@type='Boolean']"/>
    </xsl:template>
    <xsl:template match="BODY">
        <xsl:apply-templates select="PROP|QUANT|PRED|CONST[@type='Boolean']"/>
    </xsl:template>
    <xsl:template match="PRED">
        <list><atom><xsl:value-of select="@name"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@type='float']">
        <list><atom>floating<xsl:value-of select="@name"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@type='alloc' or @type='lock']">
        <list><atom>alloc<xsl:value-of select="@name"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='LT']">
        <list><atom>&lt;</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='LE']">
        <list><atom>&lt;=</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='LE' and @type='type']">
        <list><atom>&lt;:</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='GT']">
        <list><atom>&gt;</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='GE']">
        <list><atom>&gt;=</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="PRED[@name='is']">
        <list><atom>EQ</atom> <atom>|@true|</atom> <list><atom>is</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list></list>
    </xsl:template>
    <xsl:template match="PRED[@name='isAllocated']">
        <list><atom>EQ</atom> <atom>|@true|</atom> <list><atom>isAllocated</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list></list>
    </xsl:template>
    <xsl:template match="TERM">
        <list><atom><xsl:value-of select="@name"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="TERM[@name='ADD']">
        <list><atom>+</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="TERM[@name='SUB']">
        <list><atom>-</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="TERM[@name='MULT']">
        <list><atom>*</atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="TERM[@type='int']">
        <list><atom>integral<xsl:value-of select="substring(@name, 0, 3)"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="TERM[@type='float']">
        <list><atom>floating<xsl:value-of select="substring(@name, 0 ,3)"/></atom> <xsl:apply-templates select="TERM|VAR|CONST"/></list>
    </xsl:template>
    <xsl:template match="VAR">
        <atom><xsl:value-of select="@name"/></atom>
    </xsl:template>
    <xsl:template match="CONST">
        <atom><xsl:value-of select="@value"/></atom>
    </xsl:template>
    <xsl:template match="CONST[@type='Boolean']">
        <atom>|@<xsl:value-of select="@value"/>|</atom>
    </xsl:template>
    <xsl:template match="list" mode="sterm">
        (<xsl:apply-templates mode="sterm"/>)
    </xsl:template>
    <xsl:template match="atom" mode="sterm">
        <xsl:value-of select="."/>
    </xsl:template>
</xsl:stylesheet>
