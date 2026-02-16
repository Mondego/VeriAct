package test.nested_classes;

class TopLevelType {

  // binary name should be test.nested_classes.TopLevelType$MemberType
  // while canonical name is test.nested_classes.TopLevelType.MemberType
  class MemberType {
    MemberType mt;
    TopLevelType.MemberType tltmt;
    test.nested_classes.TopLevelType tlt;
    test.nested_classes.TopLevelType.MemberType canon_tltmt;

    class SecondNest {
    }
  }
}