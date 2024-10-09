use std::collections::HashMap;

use crate::triangle::Triangle;
use nalgebra::{Matrix4, Vector3, Vector4};

use std::cmp::{min, max};

#[allow(dead_code)]
pub enum Buffer {
    Color,
    Depth,
    Both,
}

#[allow(dead_code)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Default, Clone, PartialEq)]
pub enum AAType {
    #[default]
    WithoutAA,
    MSAA,
    TAA,
}

#[derive(Default, Clone)]
pub struct Rasterizer {
    AA: AAType,

    model: Matrix4<f64>,
    view: Matrix4<f64>,
    projection: Matrix4<f64>,
    pos_buf: HashMap<usize, Vec<Vector3<f64>>>,
    ind_buf: HashMap<usize, Vec<Vector3<usize>>>,
    col_buf: HashMap<usize, Vec<Vector3<f64>>>,

    frame_buf: Vec<Vector3<f64>>,

    // used by WithoutAA only
    depth_buf: Vec<f64>,

    // used by MSAA and TAA
    frame_sample: Vec<Vector3<f64>>,
    depth_sample: Vec<f64>,

    width: u64,
    height: u64,
    next_id: usize,

    taa_first: bool,
    taa_cnt: usize,
}

#[derive(Clone, Copy)]
pub struct PosBufId(usize);

#[derive(Clone, Copy)]
pub struct IndBufId(usize);

#[derive(Clone, Copy)]
pub struct ColBufId(usize);

impl Rasterizer {
    pub fn new(w: u64, h: u64, aa: AAType) -> Self {
        let mut r = Rasterizer::default();
        r.AA = aa;
        r.taa_cnt = 0 as usize;
        r.taa_first = true;

        r.width = w;
        r.height = h;
        r.frame_buf.resize((w * h) as usize, Vector3::zeros());
        r.depth_buf.resize((w * h) as usize, 0.0);
        r.frame_sample
            .resize((w * h * 4) as usize, Vector3::zeros());
        r.depth_sample.resize((w * h * 4) as usize, 0.0);
        r
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        ((self.height - 1 - y as u64) * self.width + x as u64) as usize
    }

    fn set_pixel_WithoutAA(&mut self, point: &Vector3<f64>, color: &Vector3<f64>) {
        let ind = (self.height as f64 - 1.0 - point.y) * self.width as f64 + point.x;
        self.frame_buf[ind as usize] = *color;
    }

    fn set_pixel_MSAA(&mut self, x: f64, y: f64) {
        let ind = self.get_index(x as usize, y as usize);
        self.frame_buf[ind] = (self.frame_sample[ind * 4]
            + self.frame_sample[ind * 4 + 1]
            + self.frame_sample[ind * 4 + 2]
            + self.frame_sample[ind * 4 + 3])
            / 4.0;
    }

    fn set_pixel_TAA(&mut self, x: f64, y: f64) {
        let ind = self.get_index(x as usize, y as usize);
        let maxium = if self.taa_first { self.taa_cnt } else { 4 };
        for i in 0..maxium {
            self.frame_buf[ind] += self.frame_sample[ind * 4 + i];
        }
        self.frame_buf[ind] /= maxium as f64;
        // println!("{}", self.frame_buf[ind]);
    }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));

                if self.AA != AAType::TAA || self.depth_sample[0] == 0.0 {
                    self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
                }
            }
            Buffer::Depth => {
                self.depth_buf.fill(f64::MIN);

                if self.AA != AAType::TAA || self.depth_sample[0] == 0.0 {
                    self.depth_sample.fill(f64::MIN);
                }
            }
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MIN);

                if self.AA != AAType::TAA || self.depth_sample[0] == 0.0 {
                    self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
                    self.depth_sample.fill(f64::MIN);
                }
            }
        }
    }

    pub fn set_model(&mut self, model: Matrix4<f64>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f64>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f64>) {
        self.projection = projection;
    }

    fn get_next_id(&mut self) -> usize {
        let res = self.next_id;
        self.next_id += 1;
        res
    }

    pub fn load_position(&mut self, positions: &Vec<Vector3<f64>>) -> PosBufId {
        let id = self.get_next_id();
        self.pos_buf.insert(id, positions.clone());
        PosBufId(id)
    }

    pub fn load_indices(&mut self, indices: &Vec<Vector3<usize>>) -> IndBufId {
        let id = self.get_next_id();
        self.ind_buf.insert(id, indices.clone());
        IndBufId(id)
    }

    pub fn load_colors(&mut self, colors: &Vec<Vector3<f64>>) -> ColBufId {
        let id = self.get_next_id();
        self.col_buf.insert(id, colors.clone());
        ColBufId(id)
    }

    pub fn draw(
        &mut self,
        pos_buffer: PosBufId,
        ind_buffer: IndBufId,
        col_buffer: ColBufId,
        _typ: Primitive,
    ) {
        let buf = &self.clone().pos_buf[&pos_buffer.0];
        let ind: &Vec<Vector3<usize>> = &self.clone().ind_buf[&ind_buffer.0];
        let col = &self.clone().col_buf[&col_buffer.0];

        let f1 = (50.0 - 0.1) / 2.0;
        let f2 = (50.0 + 0.1) / 2.0;

        let mvp = self.projection * self.view * self.model;

        for i in ind {
            let mut t = Triangle::new();
            let mut v = vec![
                mvp * to_vec4(buf[i[0]], Some(1.0)), // homogeneous coordinates
                mvp * to_vec4(buf[i[1]], Some(1.0)),
                mvp * to_vec4(buf[i[2]], Some(1.0)),
            ];

            for vec in v.iter_mut() {
                *vec = *vec / vec.w;
            }
            for vert in v.iter_mut() {
                vert.x = 0.5 * self.width as f64 * (vert.x + 1.0);
                vert.y = 0.5 * self.height as f64 * (vert.y + 1.0);
                vert.z = vert.z * f1 + f2;
            }
            for j in 0..3 {
                // t.set_vertex(j, Vector3::new(v[j].x, v[j].y, v[j].z));
                t.set_vertex(j, v[j]);
                t.set_vertex(j, v[j]);
                t.set_vertex(j, v[j]);
            }
            let col_x = col[i[0]];
            let col_y = col[i[1]];
            let col_z = col[i[2]];
            t.set_color(0, col_x[0], col_x[1], col_x[2]);
            t.set_color(1, col_y[0], col_y[1], col_y[2]);
            t.set_color(2, col_z[0], col_z[1], col_z[2]);

            match self.AA {
                AAType::WithoutAA => self.rasterize_triangle_WithoutAA(&t),
                AAType::MSAA => self.rasterize_triangle_MSAA(&t),
                AAType::TAA => self.rasterize_triangle_TAA(&t),
            }
        }

        if self.AA == AAType::TAA {
            self.taa_cnt += 1;
            if self.taa_cnt == 4 {
                self.taa_cnt = 0;
                self.taa_first = false;
            }
        }

        match self.AA {
            AAType::WithoutAA => (),
            AAType::MSAA => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        self.set_pixel_MSAA(x as f64, y as f64);
                    }
                }
            }
            AAType::TAA => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        self.set_pixel_TAA(x as f64, y as f64);
                    }
                }
            }
        }
    }

    pub fn rasterize_triangle_WithoutAA(&mut self, t: &Triangle) {
        let mut v: [Vector3<f64>; 3] = [Vector3::new(0.0, 0.0, 0.0); 3];
        for tmp in 0..3 {
            v[tmp] = Vector3::new(
                t.v[tmp].x / t.v[tmp].w,
                t.v[tmp].y / t.v[tmp].w,
                t.v[tmp].z / t.v[tmp].w,
            );
        }
        let mut small_y: u64 = min(v[0].y.round() as u64, min(v[1].y.round() as u64, v[2].y.round() as u64));
        small_y = max(small_y, 1) - 1;
        let mut big_y: u64 = max(v[0].y.round() as u64, max(v[1].y.round() as u64, v[2].y.round() as u64));
        big_y = min(big_y, self.height - 1) + 1;

        let mut small_x: u64 = min(v[0].x.round() as u64, min(v[1].x.round() as u64, v[2].x.round() as u64));
        small_x = max(small_x, 1) - 1;
        let mut big_x: u64 = max(v[0].x.round() as u64, max(v[1].x.round() as u64, v[2].x.round() as u64));
        big_x = min(big_x, self.width - 1) + 1;
        
        for y in small_y..big_y {
            for x in small_x..big_x {
                let x = x as f64;
                let y = y as f64;
                if inside_triangle(x + 0.5, y + 0.5, &v) {
                    let (c0, c1, c2) = compute_barycentric2d(x + 0.5, y + 0.5, &v);
                    let full_coordinate = c0 * v[0] + c1 * v[1] + c2 * v[2];
                    // have checked at x == full_coordiate.x, y == full_coordinate.y
                    // println!("x - full_coordinate.x = {}", x as f64 - full_coordinate.x);
                    // println!("y - full_coordinate.y = {}", y as f64 - full_coordinate.y);
                    let ind: usize = self.get_index(x as usize, y as usize);
                    if self.depth_buf[ind] < full_coordinate.z {
                        self.depth_buf[ind] = full_coordinate.z;
                        self.set_pixel_WithoutAA(&Vector3::new(x, y, 0.0), &(t.color[0] * 255.0));
                    }
                }
            }
        }
    }

    // note that since the color is the same in every triangle
    // there is no difference betwwen SSAA and MSAA
    pub fn rasterize_triangle_MSAA(&mut self, t: &Triangle) {
        let mut v: [Vector3<f64>; 3] = [Vector3::new(0.0, 0.0, 0.0); 3];
        for tmp in 0..3 {
            v[tmp] = Vector3::new(
                t.v[tmp].x / t.v[tmp].w,
                t.v[tmp].y / t.v[tmp].w,
                t.v[tmp].z / t.v[tmp].w,
            );
        }
        let moveX = [-0.25, 0.25, 0.25, -0.25];
        let moveY = [-0.25, -0.25, 0.25, 0.25];
        
        let mut small_y: u64 = min(v[0].y.round() as u64, min(v[1].y.round() as u64, v[2].y.round() as u64));
        small_y = max(small_y, 1) - 1;
        let mut big_y: u64 = max(v[0].y.round() as u64, max(v[1].y.round() as u64, v[2].y.round() as u64));
        big_y = min(big_y, self.height - 1) + 1;

        let mut small_x: u64 = min(v[0].x.round() as u64, min(v[1].x.round() as u64, v[2].x.round() as u64));
        small_x = max(small_x, 1) - 1;
        let mut big_x: u64 = max(v[0].x.round() as u64, max(v[1].x.round() as u64, v[2].x.round() as u64));
        big_x = min(big_x, self.width - 1) + 1;
        
        for y in small_y..big_y {
            for x in small_x..big_x {
                let x = x as f64;
                let y = y as f64;
                for i in 0..4 {
                    let sx = x + 0.5 + moveX[i];
                    let sy = y + 0.5 + moveY[i];
                    if inside_triangle(sx, sy, &v) {
                        let (c0, c1, c2) = compute_barycentric2d(sx, sy, &v);
                        let full_coordinate = c0 * v[0] + c1 * v[1] + c2 * v[2];
                        let ind: usize = self.get_index(x as usize, y as usize) * 4 + i;
                        if self.depth_sample[ind] < full_coordinate.z {
                            self.depth_sample[ind] = full_coordinate.z;
                            self.frame_sample[ind] = t.color[0] * 255.0;
                        }
                    }
                }
            }
        }
    }

    pub fn rasterize_triangle_TAA(&mut self, t: &Triangle) {
        let mut v: [Vector3<f64>; 3] = [Vector3::new(0.0, 0.0, 0.0); 3];
        for tmp in 0..3 {
            v[tmp] = Vector3::new(
                t.v[tmp].x / t.v[tmp].w,
                t.v[tmp].y / t.v[tmp].w,
                t.v[tmp].z / t.v[tmp].w,
            );
        }
        let moveX = [-0.25, 0.25, 0.25, -0.25];
        let moveY = [-0.25, -0.25, 0.25, 0.25];

        let mut small_y: u64 = min(v[0].y.round() as u64, min(v[1].y.round() as u64, v[2].y.round() as u64));
        small_y = max(small_y, 1) - 1;
        let mut big_y: u64 = max(v[0].y.round() as u64, max(v[1].y.round() as u64, v[2].y.round() as u64));
        big_y = min(big_y, self.height - 1) + 1;

        let mut small_x: u64 = min(v[0].x.round() as u64, min(v[1].x.round() as u64, v[2].x.round() as u64));
        small_x = max(small_x, 1) - 1;
        let mut big_x: u64 = max(v[0].x.round() as u64, max(v[1].x.round() as u64, v[2].x.round() as u64));
        big_x = min(big_x, self.width - 1) + 1;
        
        for y in small_y..big_y {
            for x in small_x..big_x {
                let x = x as f64;
                let y = y as f64;
                let sx = x + 0.5 + moveX[self.taa_cnt];
                let sy = y + 0.5 + moveY[self.taa_cnt];
                if inside_triangle(sx, sy, &v) {
                    let (c0, c1, c2) = compute_barycentric2d(sx, sy, &v);
                    let full_coordinate = c0 * v[0] + c1 * v[1] + c2 * v[2];
                    let ind: usize = self.get_index(x as usize, y as usize) * 4 + self.taa_cnt;
                    if self.depth_sample[ind] < full_coordinate.z {
                        self.depth_sample[ind] = full_coordinate.z;
                        self.frame_sample[ind] = t.color[0] * 255.0;
                    }
                }
            }
        }
    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }
}

fn to_vec4(v3: Vector3<f64>, w: Option<f64>) -> Vector4<f64> {
    Vector4::new(v3.x, v3.y, v3.z, w.unwrap_or(1.0))
}

fn inside_triangle(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> bool {
    let A: Vector3<f64> = Vector3::new(v[0].x, v[0].y, 0.0);
    let B: Vector3<f64> = Vector3::new(v[1].x, v[1].y, 0.0);
    let C: Vector3<f64> = Vector3::new(v[2].x, v[2].y, 0.0);
    let P: Vector3<f64> = Vector3::new(x, y, 0.0);
    let AB: Vector3<f64> = B - A;
    let BC: Vector3<f64> = C - B;
    let CA: Vector3<f64> = A - C;
    let AP: Vector3<f64> = P - A;
    let BP: Vector3<f64> = P - B;
    let CP: Vector3<f64> = P - C;

    let sgn1: bool = (AB.cross(&AP)).z >= 0.0;
    let sgn2: bool = (BC.cross(&BP)).z >= 0.0;
    let sgn3: bool = (CA.cross(&CP)).z >= 0.0;
    if sgn1 && sgn2 && sgn3 {
        true
    } else if (!sgn1) && (!sgn2) && (!sgn3) {
        true
    } else {
        false
    }
}

fn compute_barycentric2d(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> (f64, f64, f64) {
    let c1 = (x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * y + v[1].x * v[2].y - v[2].x * v[1].y)
        / (v[0].x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * v[0].y + v[1].x * v[2].y
            - v[2].x * v[1].y);
    let c2 = (x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * y + v[2].x * v[0].y - v[0].x * v[2].y)
        / (v[1].x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * v[1].y + v[2].x * v[0].y
            - v[0].x * v[2].y);
    let c3 = (x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * y + v[0].x * v[1].y - v[1].x * v[0].y)
        / (v[2].x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * v[2].y + v[0].x * v[1].y
            - v[1].x * v[0].y);
    (c1, c2, c3)
}
