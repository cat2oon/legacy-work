//
// Three.js Editor Scripts
// For Checking Face Position
//
// 최종 체크할 때 MR 없이 R만으로 체크할 것
//

let rotationFlag = false;
let I = new THREE.Matrix4();
let MR = new THREE.Matrix4();
MR.set(-1, 0, 0, 0, 0, 1, 0, 0,0, 0, 1, 0, 0, 0, 0, 1);
let ZINV = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), Math.PI);
let Y_ROT = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), Math.PI / 1.2);

function update(event) {
	if (rotationFlag)
		return;	

    // 회전 변환 행렬
    let R = new THREE.Matrix4();
    let r1 =  [ -0.836, 0.530, -0.143, ];
    let r2 =  [ 0.535, 0.845, 0.006,  ];
    let r3 =  [ 0.124, -0.072, -0.990     ];

    R.set(r1[0], r1[1], r1[2],  0,
             r2[0], r2[1], r2[2],  0,
             r3[0], r3[1], r3[2],  0,
             0,    0,    0,      1);

    // let MR = new THREE.Matrix4();
    // MR.set(-1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 0, 1);

    // 회전 행렬 적용
    this.applyMatrix(R);
    //this.applyMatrix(ZINV);
    // this.applyMatrix(MR.multiply(R));
	
	rotationFlag = true;
}


// YZ Plane Reflect Rotation
// let MR = new THREE.Matrix4();
// MR.set(-1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 0, 1);

/*
let r11 =  0.976;
let r12 =  0.025;
let r13 =  0.215;
let r21 = 0.101;
let r22 = 0.827;
let r23 = -0.553;
let r31 = -0.192;
let r32 = -0.562;
let r33 = -0.805;

R.set(r11, r12, r13, 0,
	  r21, r22, r23, 0,
	  r31, r32, r33, 0,
		0,   0,   0, 1); 

this.applyMatrix(MR.multiply(R));
/*

