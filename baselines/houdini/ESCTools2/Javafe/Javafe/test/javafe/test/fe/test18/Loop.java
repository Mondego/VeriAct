// One of these must generate an error...

class Loop extends Loop1 {}

class Loop1 extends Loop2 {}

class Loop2 extends Loop {}
