interface Base extends Middle {}

interface Middle extends Top, Middle, Top {}	// error (crashes javac)

interface Top {}
