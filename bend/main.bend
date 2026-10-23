# Groot 1 — native command-line engine (CPU threads / GPU via Bend's runtime).
#
#   ./groot model.gbw req.bin --threads 16 --gpu off
#
# bend/groot/layout.bend must be generated from the same model.gbw (groot1.bendio.layout_from_gbw);
# it supplies the weight data offset and size. The request blob's header supplies its own size.
# Prints the output vector [probs ; confidence logits ; logits], one value per line.
import Base
import ./groot/groot.bend as G
import ./groot/kernels.bend as K
import ./groot/layout.bend as L

def Blob.load_f32(path: String, offset: U32, depth: U32) -> IO(Array<F32>):
  import "./effs/blob_load_f32.c"
  import "./effs/blob_load_f32.js"

def Blob.load_u32(path: String, offset: U32, depth: U32) -> IO(Array<U32>):
  import "./effs/blob_load_u32.c"
  import "./effs/blob_load_u32.js"

def show.go(k: Nat, +i: U32, r: Array<F32> & F32) -> IO(Unit):
  match k:
    case 0n:
      (a, v) = r
      IO.print(F32.show(v))
    case 1n+p:
      (a, v) = r
      +j = (i + 1 : U32)
      do IO<Unit>:
        IO.print(F32.show(v))
        show.go(p, j, Array.get(F32, a, j))

def show(out: Array<F32>, +n: U32) -> IO(Unit):
  show.go(U32.to_nat(U32.sub(n, 1)), 0, Array.get(F32, out, 0))

# the first n+1 words of an array, as a list
def head.go(n: Nat, +i: U32, r: Array<U32> & U32) -> List<U32>:
  match n:
    case 0n:
      (a, v) = r
      [v]
    case 1n+p:
      (a, v) = r
      +j = (i + 1 : U32)
      Con{v, head.go(p, j, Array.get(U32, a, j))}

# (words in the blob, output values) from a request header (groot1.bendio.RequestBlob):
# n_chunks, chunk_len_max, n_questions, q_len_max, k_max, n_options
def sizes.of(+nch: U32, +clen: U32, +nq: U32, +qlen: U32, +kmax: U32, +nopt: U32) -> U32 & U32:
  ((6 + nch + (4 * nq) + (nch * clen) + (3 * nq * qlen) + (nq * kmax) : U32), (2 * nopt + nq : U32))

def sizes(h: List<U32>) -> U32 & U32:
  match h:
    case Con{nch, Con{clen, Con{nq, Con{qlen, Con{kmax, Con{nopt, rest}}}}}}:
      sizes.of(nch, clen, nq, qlen, kmax, nopt)
    case _:
      (0, 0)

def req_sizes(h: Array<U32>) -> U32 & U32:
  sizes(head.go(5n, 0, Array.get(U32, h, 0)))

def answer(w: Array<F32>, bpath: String, n: U32 & U32) -> IO(Unit):
  (words, nout) = n
  do IO<Unit>:
    b : Array<U32> <- Blob.load_u32(bpath, 0, U32.from_nat(K.depth_for(words)))
    t0 : Nat <- IO.now()
    out : Array<F32> = G.forward(w, b)
    show(out, nout)
    t1 : Nat <- IO.now()
    IO.print_err("groot forward ms = " ++ Nat.show(Nat.sub(t1, t0)))

def run(wpath: String, +bpath: String) -> IO(Unit):
  do IO<Unit>:
    w : Array<F32> <- Blob.load_f32(wpath, L.data_offset(), L.weight_depth())
    h : Array<U32> <- Blob.load_u32(bpath, 0, 3)
    answer(w, bpath, req_sizes(h))

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{wpath, Con{bpath, Nil{}}}}:
      run(wpath, bpath)
    case _:
      do IO<Unit>:
        IO.print_err("usage: groot WEIGHTS.gbw REQUEST.bin [--threads N] [--gpu off|HEAP]")
        IO.exit(2)

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
