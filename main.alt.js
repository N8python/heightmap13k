 // Get the canvas and context
 var canvas = document.getElementById("canvas");
 canvas.width = window.innerWidth;
 canvas.height = window.innerHeight;
 var gl = canvas.getContext("webgl2");
 var ext = gl.getExtension('EXT_color_buffer_half_float');
 // Create the shaders
 var shaderSources = {
     vertex: `#version 300 es
        precision highp float;in vec2 aPosition;out vec2 vPosition;void main(){vPosition = aPosition;gl_Position = vec4(aPosition, 0, 1);}
    `,
     fragmentMain: file("main-shader-min.frag"),
     fragmentHeightMap: file("height-shader-min.frag")
 };

 var shaders = {
     vertex: compileShader(gl, 35633, shaderSources.vertex),
     fragmentMain: compileShader(gl, 35632, shaderSources.fragmentMain),
     fragmentHeightMap: compileShader(gl, 35632, shaderSources.fragmentHeightMap),
 };

 // Create the programs
 var programs = {
     main: createProgram(gl, shaders.vertex, shaders.fragmentMain),
     heightMap: createProgram(gl, shaders.vertex, shaders.fragmentHeightMap),
 };
 // Create the buffer
 var positionBuffer = createBuffer(gl, [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, ]);

 // Create the textures
 var heightTexture = createTexture(gl, 6403, 2048, 2048, 5121, 33321);

 // Create the framebuffers
 var heightFBO = createFramebuffer(gl, heightTexture);

 // Clear the canvas and render
 gl.clearColor(0, 0, 0, 0);


 renderToTexture(gl, programs.heightMap, heightFBO, positionBuffer, [2048, 2048]);
 gl.bindTexture(3553, heightTexture);
 gl.texParameteri(3553, 10241, 9987);
 gl.texParameteri(3553, 10240, 9729);
 gl.generateMipmap(3553);
 const keys = {};
 document.addEventListener('keydown', (e) => {
     keys[e.key] = true;
 });
 document.addEventListener('keyup', (e) => {
     keys[e.key] = false;
 });
 document.addEventListener('click', e => {
     canvas.requestPointerLock();
 })
 document.addEventListener('mousemove', e => {
     if (document.pointerLockElement === canvas) {
         xRotation += e.movementX * 0.001;
         yRotation -= e.movementY * 0.001;
         yRotation = Math.max(Math.min(yRotation, Math.PI / 2), -Math.PI / 2);
     }
 })
 let cameraPos = [0, 20, 0];
 let xRotation = 0;
 let yRotation = 0;

 function render() {
     let cameraDir = [0, 0, 0]
     cameraDir[0] = Math.cos(xRotation) * Math.cos(yRotation);
     cameraDir[1] = Math.sin(yRotation);
     cameraDir[2] = Math.sin(xRotation) * Math.cos(yRotation);
     const forward = cameraDir;
     // We don't have a library, so we have to do the math ourselves
     const right = [
         Math.cos(xRotation + Math.PI / 2) * Math.cos(yRotation),
         Math.sin(yRotation),
         Math.sin(xRotation + Math.PI / 2) * Math.cos(yRotation)
     ];
     if (keys['w']) {
         cameraPos[0] += forward[0];
         cameraPos[1] += forward[1];
         cameraPos[2] += forward[2];
     }
     if (keys['s']) {
         cameraPos[0] -= forward[0];
         cameraPos[1] -= forward[1];
         cameraPos[2] -= forward[2];
     }
     if (keys['a']) {
         cameraPos[0] -= right[0];
         cameraPos[1] -= right[1];
         cameraPos[2] -= right[2];
     }
     if (keys['d']) {
         cameraPos[0] += right[0];
         cameraPos[1] += right[1];
         cameraPos[2] += right[2];
     }

     // Pass the color and height textures to the main program
     gl.useProgram(programs.main);
     var heightTextureLocation = gl.getUniformLocation(programs.main, 'heightTexture');
     var timeLocation = gl.getUniformLocation(programs.main, 'time');
     var posLocation = gl.getUniformLocation(programs.main, 'cameraPos');
     var dirLocation = gl.getUniformLocation(programs.main, 'cameraDir');
     gl.uniform1i(heightTextureLocation, 0); // texture unit 1

     gl.uniform3fv(posLocation, cameraPos);

     gl.uniform3fv(dirLocation, cameraDir);

     const resolution = [canvas.width, canvas.height];
     gl.uniform2fv(gl.getUniformLocation(programs.main, 'resolution'), resolution);

     gl.activeTexture(33984);
     gl.bindTexture(3553, heightTexture);

     gl.uniform1f(timeLocation, performance.now() / 1000);

     renderToScreen(gl, programs.main, positionBuffer);

     // Loop at 60fps
     requestAnimationFrame(render);
 }

 // Start rendering
 render();

 // Helper functions
 function compileShader(gl, type, source) {
     var shader = gl.createShader(type);
     gl.shaderSource(shader, source);
     gl.compileShader(shader);
     if (!gl.getShaderParameter(shader, 35713)) {
         console.error('Shader failed to compile: ' + gl.getShaderInfoLog(shader));
     }
     return shader;
 }

 function createProgram(gl, vertexShader, fragmentShader) {
     var program = gl.createProgram();
     gl.attachShader(program, vertexShader);
     gl.attachShader(program, fragmentShader);
     gl.linkProgram(program);
     if (!gl.getProgramParameter(program, 35714)) {
         console.error('Program failed to link: ' + gl.getProgramInfoLog(program));
     }
     return program;
 }

 function createBuffer(gl, data) {
     var buffer = gl.createBuffer();
     gl.bindBuffer(34962, buffer);
     gl.bufferData(34962, new Float32Array(data), 35044);
     return buffer;
 }

 function createTexture(gl, format, width, height, type, internalFormat = format) {
     var texture = gl.createTexture();
     gl.bindTexture(3553, texture);
     if (gl.getError() != 0) {
         console.error("Error binding texture");
     }

     gl.texImage2D(3553, 0, internalFormat, width, height, 0, format, type, null);
     if (gl.getError() != 0) {
         console.error("Error setting texture image");
     }
     gl.texParameteri(3553, 10241, 9987);
     gl.texParameteri(3553, 10240, 9729);

     // Repeat
     gl.texParameteri(3553, 10242, 10497);
     gl.texParameteri(3553, 10243, 10497);

     return texture;
 }

 function createFramebuffer(gl, texture) {
     var fbo = gl.createFramebuffer();
     gl.bindFramebuffer(36160, fbo);
     gl.framebufferTexture2D(36160, 36064, 3553, texture, 0);
     gl.texParameteri(3553, 10241, 9729)
     return fbo;
 }

 function renderToTexture(gl, program, fbo, positionBuffer, size) {
     gl.bindFramebuffer(36160, fbo);
     gl.viewport(0, 0, size[0], size[1]);
     renderWithProgram(gl, program, positionBuffer);
 }

 function renderToScreen(gl, program, positionBuffer) {
     gl.bindFramebuffer(36160, null);
     gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
     renderWithProgram(gl, program, positionBuffer);
 }

 function renderWithProgram(gl, program, positionBuffer) {
     gl.useProgram(program);
     gl.enableVertexAttribArray(0);
     gl.bindBuffer(34962, positionBuffer);
     gl.vertexAttribPointer(0, 2, 5126, false, 0, 0);
     gl.drawArrays(4, 0, 6);
 }