import { spglslAngleCompile } from "spglsl";
import fs from "fs";
const result = await spglslAngleCompile({
    mainFilePath: "example_shader.frag",
    mainSourceCode: fs.readFileSync("example_shader.frag", "utf8"),

    minify: true,

    // Mangle everything, except uniforms and globals, "main" and function starting with "main"
    mangle: true,
    compileMode: "Optimize",

});
result.output = result.output.replace(/normalize/g, "n");
result.output = "#define normalize n\n" + result.output;
fs.writeFileSync("example_out.frag", result.output);