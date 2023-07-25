import fs from 'fs';
import * as CONSTANTS from "webgl-constants";


// Match for the regex: gl.(GL_{CONSTANT})
// Replace with the value of the constant in CONSTANTS[GL_{CONSTANT}]
let minifiedJS = fs.readFileSync("main.js").toString();

minifiedJS = minifiedJS.replace(/gl\.([A-Z0-9_]+)/g, (match, p1) => {

    console.log(match, p1, CONSTANTS["GL_" + p1])
    return CONSTANTS["GL_" + p1];
});

fs.writeFileSync("main.alt.js", minifiedJS);