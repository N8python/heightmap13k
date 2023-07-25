import fs from 'fs';
import * as CONSTANTS from "webgl-constants";
let minifiedJS = fs.readFileSync("main.min.js").toString();

// Match for the regex: file("path.frag")
// Replace with the contents of path.frag
minifiedJS = minifiedJS.replace(/file\("(.*?)"\)/g, (match, p1) => {
    return "`" + fs.readFileSync(p1).toString() + "`";
})

// Match for gl.${CONSTANT}, then replace with the value of the constant in CONSTANTS[GL_{CONSTANT}]

fs.writeFileSync("main.min.js", minifiedJS);


const finalMinified = fs.readFileSync("main.min.js").toString();
const index = fs.readFileSync("index-proto.html").toString();
fs.writeFileSync("index.html", index.replace(/<mainscript><\/mainscript>/, `<script>${finalMinified}</script>`));