package escjava.ant;

import org.apache.tools.ant.BuildException;
import org.apache.tools.ant.Project;
import org.apache.tools.ant.taskdefs.Execute;
import org.apache.tools.ant.taskdefs.Javac;
import org.apache.tools.ant.taskdefs.LogStreamHandler;
import org.apache.tools.ant.taskdefs.compilers.DefaultCompilerAdapter;
import org.apache.tools.ant.types.Commandline;
import org.apache.tools.ant.types.Path;
import org.apache.tools.ant.util.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Define an Escj compiler adapter.  This class is only a very simple
 * wrapper and does not try to implement of the command line options
 * of escj.
 *
 * @author wsargent
 * @version $Revision$
 *
 * @since May 16, 2004
 */
public class ESCJavaTask extends DefaultCompilerAdapter
{
  private FileUtils fileUtils = FileUtils.newFileUtils();

  /* (non-Javadoc)
   * @see org.apache.tools.ant.taskdefs.compilers.DefaultCompilerAdapter#execute()
   */
  public boolean execute() throws BuildException
  {
    if (super.attributes == null)
      {
        throw new BuildException("Null attributes");
      }

    if (super.project == null)
      {
        throw new BuildException("Null project");
      }

    Javac javac = getJavac();

    if (javac == null)
      {
        throw new BuildException("null javac");
      }

    super.attributes.log("Using escj compiler", 3);

    Path classpath = new Path(super.project);

    if (super.bootclasspath != null)
      {
        classpath.append(super.bootclasspath);
      }

    classpath.addExtdirs(super.extdirs);

    if ((super.bootclasspath == null) || (super.bootclasspath.size() == 0))
      {
        super.includeJavaRuntime = true;
      }

    classpath.append(getCompileClasspath());

    if (super.compileSourcepath != null)
      {
        classpath.append(super.compileSourcepath);
      } else
        {
          classpath.append(super.src);
        }

    Commandline cmd = new Commandline();
    String exec = javac.getExecutable();
    cmd.setExecutable((exec != null) ? exec : "java");

    String simplifyProperty = super.project.getProperty(
                                                        "build.compiler.simplify");

    // Make sure the simplify argument comes before...
    if (simplifyProperty != null)
      {
        cmd.createArgument().setValue("-Dsimplify=" + simplifyProperty);
      }

    cmd.createArgument().setValue("-classpath");
    cmd.createArgument().setPath(classpath);

    cmd.createArgument().setValue("escjava.Main");

    cmd.createArgument().setValue("-classpath");
    cmd.createArgument().setPath(classpath);

    /*
      if (super.deprecation)
      {
      cmd.createArgument().setValue("-deprecation");
      }
    */
    /*
      if (super.destDir != null)
      {
      cmd.createArgument().setValue("-d");
      cmd.createArgument().setFile(super.destDir);
      }
    */
    /*
      if (super.encoding != null)
      {
      cmd.createArgument().setValue("-encoding");
      cmd.createArgument().setValue(super.encoding);
      }
    */
    /*
      if (super.debug)
      {
      cmd.createArgument().setValue("-g");
      }
    */
    /*
      if (super.optimize)
      {
      cmd.createArgument().setValue("-O");
      }
    */
    setCounterExample(cmd);
    setDepend(cmd);
    setEajava(cmd);
    setEajml(cmd);
    setEnableAssertions(cmd);
    setLoopsafe(cmd);
    setLoop(cmd);
    setNoCheck(cmd);
    setNoCautions(cmd);
    setNoRedundancy(cmd);
    setNoTrace(cmd);
    setNoWarn(cmd);
    setPxLog(cmd);
    setPlainWarning(cmd);
    setQuiet(cmd);
    setRoutine(cmd);
    setRoutineIndirect(cmd);
    setSource(cmd);
    setSourcePath(cmd);
    setSuggest(cmd);
    setSxLog(cmd);
    setTarget(cmd);
    setTypeCheck(cmd);
    setUseVars(cmd);
    setUseFcns(cmd);
    setUseVarsForModelVars(cmd);
    setUseFcnsForModelVars(cmd);
    setWarn(cmd);
    setVerbose(cmd);

    // Set the -f value to use the files following.
    cmd.createArgument().setValue("-f");

    addCurrentCompilerArgs(cmd);

    int firstFileName = cmd.size();
    logAndAddFilesToCompile(cmd);

    return executeExternalCompile(cmd.getCommandline(), firstFileName) == 0;
  }

  /**
   * @param cmd
   */
  private void setUseFcnsForModelVars(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "usefcnsformodelvars", "useFcnsForModelVars");
  }

  /**
   * @param cmd
   */
  private void setUseVarsForModelVars(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "usevarsformodelvars", "useVarsForModelVars");
  }

  /**
   * @param cmd
   */
  private void setWarn(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "warn", "warn");
  }

  /**
   * @param cmd
   */
  private void setUseFcns(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setArgument(cmd, "usefcns", "useFcns");
  }

  /**
   * @param cmd
   */
  private void setRoutineIndirect(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "routineindirect", "routineIndirect");
  }

  /**
   * @param cmd
   */
  private void setPlainWarning(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setArgument(cmd, "plainwarning", "plainwarning");
  }

  /**
   * @param cmd
   */
  private void setRoutine(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "routine";
    setTextArgument(cmd, arg, arg);
  }

  /**
   * @param cmd
   */
  private void setLoop(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "loop";
    setTextArgument(cmd, arg, arg);
  }

  /**
   * Sets the -quiet option.
   *
   * @param cmd
   */
  private void setQuiet(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "quiet";
    setArgument(cmd, arg, arg);
  }

  private void setVerbose(/*@ non_null @*/ Commandline cmd)
  {
    if (super.verbose)
      {
        cmd.createArgument().setValue("-v");
      }
  }

  private void setDepend(/*@ non_null @*/ Commandline cmd)
  {
    if (super.depend)
      {
        cmd.createArgument().setValue("-depend");
      }
  }

  private void setTarget(/*@ non_null @*/ Commandline cmd)
  {
    if (super.target != null)
      {
        cmd.createArgument().setValue("-target");
        cmd.createArgument().setValue(super.target);
      }
  }

  private void setNoWarn(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    if (super.attributes == null)
      {
        throw new BuildException("Null attributes");
      }

    if (super.attributes.getNowarn())
      {
        cmd.createArgument().setValue("-nowarn");
      }
  }

  private void setSource(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    if (super.attributes == null)
      {
        throw new BuildException("Null attributes");
      }

    if (super.attributes.getSource() != null)
      {
        cmd.createArgument().setValue("-source");
        cmd.createArgument().setValue(super.attributes.getSource());
      }
  }

  private void setSourcePath(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    if (super.attributes == null)
      {
        throw new BuildException("Null attributes");
      }

    if (super.attributes.getSourcepath() != null)
      {
        cmd.createArgument().setValue("-sourcepath");

        Path path = super.attributes.getSourcepath();
        String pathString = (path == null) ? null : path.toString();
        cmd.createArgument().setValue(pathString);
      }
  }

  private void setSxLog(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "sxlog", "sxLog");
  }

  private void setPxLog(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    setTextArgument(cmd, "pxlog", "pxLog");
  }

  private void setCounterExample(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "counterexample";
    setArgument(cmd, arg, arg);
  }

  private void setEajava(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "eajava";
    setArgument(cmd, arg, arg);
  }

  private void setEajml(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "eajml";
    setArgument(cmd, arg, arg);
  }

  private void setEnableAssertions(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "enableassertions";
    setArgument(cmd, arg, arg);
  }

  private void setUseVars(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "usevars";
    setArgument(cmd, arg, "useVars");
  }

  private void setTypeCheck(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "typecheck";
    setArgument(cmd, arg, arg);
  }

  private void setSuggest(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "suggest";
    setArgument(cmd, arg, arg);
  }

  private void setNoTrace(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "notrace";
    setArgument(cmd, arg, arg);
  }

  private void setNoRedundancy(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "noredundancy";
    setArgument(cmd, arg, arg);
  }

  private void setNoCautions(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "nocautions";
    setArgument(cmd, arg, "noCautions");
  }

  private void setNoCheck(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "nocheck";
    setArgument(cmd, arg, arg);
  }

  private void setLoopsafe(/*@ non_null @*/ Commandline cmd) throws BuildException
  {
    String arg = "loopsafe";
    setArgument(cmd, arg, arg);
  }

  /**
   * Do the compile with the specified arguments.
   *
   * @param args - arguments to pass to process on command line
   * @param firstFileName - index of the first source file in args, if the
   *        index is negative, no temporary file will ever be created, but
   *        this may hit the command line length limit on your system.
   * @param quoteFiles - if set to true, filenames containing spaces will be
   *        quoted when they appear in the external file. This is necessary
   *        when running JDK 1.4's javac and probably others.
   *
   * @return DOCUMENT ME!
   *
   * @throws BuildException DOCUMENT ME!
   *
   * @since Ant 1.6
   */
  //@ also
  //@   requires args != null;
  protected int executeExternalCompile(String[] args,
                                       int firstFileName,
                                       boolean quoteFiles)
  {
    String[] commandArray = null;
    File tmpFile = null;

    try
      {
      
        /*
         * Many system have been reported to get into trouble with
         * long command lines - no, not only Windows ;-).
         *
         * POSIX seems to define a lower limit of 4k, so use a temporary
         * file if the total length of the command line exceeds this limit.
         */
        if ((Commandline.toString(args).length() > 4096) && (firstFileName >= 0))
          {
            PrintWriter out = null;

            try
              {
                File userDir = getJavac().getTempdir();

                if (userDir == null)
                  {
                    String userDirName = System.getProperty("user.dir");
                    userDir = new File(userDirName);
                  }

                tmpFile = fileUtils.createTempFile("files", "", userDir);
                tmpFile.deleteOnExit();
                out = new PrintWriter(new FileWriter(tmpFile));

                for (int i = firstFileName; i < args.length; i++)
                  {
                    if (quoteFiles && (args[i].indexOf(" ") > -1))
                      {
                        args[i] = args[i].replace('\\', '/');
                        out.println("\"" + args[i] + "\"");
                      } else
                        {
                          out.println(args[i]);
                        }
                  }

                out.flush();
                commandArray = new String[firstFileName + 1];
                System.arraycopy(args, 0, commandArray, 0, firstFileName);
                commandArray[firstFileName] = tmpFile.toString();
              } catch (IOException e)
                {
                  throw new BuildException("Error creating temporary file",
                                           e, location);
                } finally
                  {
                    if (out != null)
                      {
                        try
                          {
                            out.close();
                          } catch (Throwable t)
                            {
                              // ignore
                            }
                      }
                  }
          } else
            {
              commandArray = args;
            }

        try
          {
            Execute exe = new Execute(new LogStreamHandler(attributes,
                                                           Project.MSG_INFO, Project.MSG_WARN));
            exe.setAntRun(project);
            exe.setWorkingDirectory(project.getBaseDir());
            exe.setCommandline(commandArray);
            exe.execute();

            return exe.getExitValue();
          } catch (IOException e)
            {
              throw new BuildException("Error running " + args[0] +
                                       " compiler", e, location);
            }
      } finally
        {
          if (tmpFile != null)
            {
              tmpFile.delete();
            }
        }
  }

  /**
   * Sets an argument with a text value from a property.
   *
   * @param cmd
   * @param pPropertyName
   * @param pArgumentName
   *
   * @throws BuildException if one of the arguments is null.
   */
  private void setTextArgument(/*@ non_null @*/ Commandline cmd,
                               /*@ non_null @*/ String pPropertyName,
                               /*@ non_null @*/ String pArgumentName) throws BuildException
  {
    if (cmd == null)
      {
        throw new BuildException("null cmd");
      }

    if (pPropertyName == null)
      {
        throw new BuildException("null pName");
      }

    if (pArgumentName == null)
      {
        throw new BuildException("null pArgument");
      }
        
    if (super.project == null)
      {
        throw new BuildException("null project");
      }

    String propertyValue = super.project.getProperty("build.compiler." +
                                                     pPropertyName);

    if (propertyValue != null)
      {
        cmd.createArgument().setValue("-" + pArgumentName + "=" +
                                      propertyValue);
      }
  }

  /**
   * Sets an argument on the commandline from a boolean property.
   *
   * @param cmd
   * @param pPropertyName
   * @param pArgumentName
   *
   * @throws BuildException DOCUMENT ME!
   */
  private void setArgument(/*@ non_null @*/ Commandline cmd,
                           /*@ non_null @*/ String pPropertyName,
                           /*@ non_null @*/ String pArgumentName) throws BuildException
  {
    if (cmd == null)
      {
        throw new BuildException("null cmd");
      }

    if (pPropertyName == null)
      {
        throw new BuildException("null pName");
      }

    if (pArgumentName == null)
      {
        throw new BuildException("null pArgument");
      }
        
    if (super.project == null)
      {
        throw new BuildException("null super.project");
      }

    String propertyName = super.project.getProperty("build.compiler." +
                                                    pPropertyName);

    if ((propertyName != null) && Project.toBoolean(propertyName))
      {
        cmd.createArgument().setValue("-" + pArgumentName);
      }
  }
}
