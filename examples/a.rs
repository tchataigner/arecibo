//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the `MinRoot` function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the `MinRoot` function per step of Nova's recursion.
use arecibo::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    Engine, Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField, PrimeFieldBits};
use flate2::{write::ZlibEncoder, Compression};
use halo2curves::bn256::Bn256;
use num_bigint::BigUint;
use std::marker::PhantomData;
use std::time::Instant;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};
use tracing_texray::TeXRayLayer;

/*******************************************************
 * Our side
 *******************************************************/
trait ChunkStepCircuit<F: PrimeField> {
  fn new() -> Self;

  fn arity() -> usize;

  fn chunk_synthesize<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
    chunk_in: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;
}

trait ChunkCircuit {
  fn new<F: PrimeField, C: ChunkStepCircuit<F>>(
    z0_primary: &[F],
    intermediate_steps_input: &[F],
  ) -> Self;
  fn initial_input(&self);
  fn num_fold_steps(&self) -> usize;
}

struct FoldStep<F: PrimeField, C: ChunkStepCircuit<F> + Clone, const N: usize> {
  circuit: C,
  next_input: [F; N],
}

impl<F: PrimeField, C: ChunkStepCircuit<F> + Clone, const N: usize> FoldStep<F, C, N> {
  pub fn new(circuit: C, inputs: [F; N]) -> Self {
    Self {
      circuit,
      next_input: inputs,
    }
  }
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> Clone for FoldStep<F, C, N> {
  fn clone(&self) -> Self {
    FoldStep {
      circuit: self.circuit.clone(),
      next_input: self.next_input.clone(),
    }
  }
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> StepCircuit<F> for FoldStep<F, C, N> {
  fn arity(&self) -> usize {
    N + self.circuit.arity()
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let (z_in, chunk_in) = z.split_at(self.circuit.arity());
    let mut z_out = self.circuit.chunk_synthesize(cs, z_in, chunk_in)?;
    z_out.extend(self.next_input.iter());
    Ok(z_out)
  }
}

struct Circuit<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> {
  circuits: Vec<FoldStep<F, C, N>>,
  num_fold_steps: usize,
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> ChunkCircuit for Circuit<F, C, N> {
  fn new(z0_primary: &[F], intermediate_steps_input: &[F]) -> Self {
    assert_eq!(
      intermediate_steps_input.len() % N,
      0,
      "intermediate_steps_input must be a multiple of N"
    );
    Self {
      circuits: C::new(),
      num_fold_steps: intermediate_steps_input.len() / N,
    }
  }

  fn initial_input(&self) -> Option<&FoldStep<F, C, N>> {
    self.circuits.get(0)
  }

  fn num_fold_steps(&self) -> usize {
    self.num_fold_steps
  }
}

/*******************************************************
 * User side
 *******************************************************/

struct ChunkStep<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F: PrimeField> ChunkStepCircuit<F> for ChunkStep<F> {
  fn new() -> Self {
    Self {
      _p: Default::default(),
    }
  }

  fn arity() -> usize {
    1
  }

  fn chunk_synthesize<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
    chunk_in: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let mut acc = z[0].clone();

    for (i, elem) in chunk_in.iter().enumerate() {
      // TODO i is not what we want here. Should be fold_step + i
      acc = acc.add(&mut cs.namespace(|| format!("add{i}")), &elem)?;
    }

    Ok(vec![acc])
  }
}

/// cargo run --release --example a
fn main() {
  let subscriber = Registry::default()
    .with(fmt::layer().pretty())
    .with(EnvFilter::from_default_env())
    .with(TeXRayLayer::new());
  tracing::subscriber::set_global_default(subscriber).unwrap();

  println!("Sketch Chunk proving pattern");
  println!("=========================================================");

  const num_iters_per_step: usize = 3;

  type C1 = Circuit<<Bn256EngineKZG as Engine>::GE, ChunkStep<_>, num_iters_per_step>;
  type E1 = Bn256EngineKZG;
  type E2 = GrumpkinEngine;
  type EE1 = arecibo::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
  type EE2 = arecibo::provider::ipa_pc::EvaluationEngine<E2>;
  type S1 = arecibo::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
  type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

  // number of iterations of MinRoot per Nova's recursive step
  let circuit_primary = C1::default();

  let circuit_secondary = TrivialCircuit::default();

  println!("Proving {num_iters_per_step} iterations of MinRoot per step");

  // produce public parameters
  let start = Instant::now();
  println!("Producing public parameters...");
  let pp = PublicParams::<E1>::setup(
    &circuit_primary,
    &circuit_secondary,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
  );
  println!("PublicParams::setup, took {:?} ", start.elapsed());

  println!(
    "Number of constraints per step (primary circuit): {}",
    pp.num_constraints().0
  );
  println!(
    "Number of constraints per step (secondary circuit): {}",
    pp.num_constraints().1
  );

  println!(
    "Number of variables per step (primary circuit): {}",
    pp.num_variables().0
  );
  println!(
    "Number of variables per step (secondary circuit): {}",
    pp.num_variables().1
  );
  /*
  // produce non-deterministic advice
  let (z0_primary, minroot_iterations) = MinRootIteration::<<E1 as Engine>::GE>::new(
    num_iters_per_step * num_steps,
    &<E1 as Engine>::Scalar::zero(),
    &<E1 as Engine>::Scalar::one(),
  );
  let minroot_circuits = (0..num_steps)
    .map(|i| MinRootCircuit {
      seq: (0..num_iters_per_step)
        .map(|j| MinRootIteration {
          x_i: minroot_iterations[i * num_iters_per_step + j].x_i,
          y_i: minroot_iterations[i * num_iters_per_step + j].y_i,
          x_i_plus_1: minroot_iterations[i * num_iters_per_step + j].x_i_plus_1,
          y_i_plus_1: minroot_iterations[i * num_iters_per_step + j].y_i_plus_1,
        })
        .collect::<Vec<_>>(),
    })
    .collect::<Vec<_>>();

  let z0_secondary = vec![<E2 as Engine>::Scalar::zero()];

  type C1 = MinRootCircuit<<E1 as Engine>::GE>;
  type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;
  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");
  let mut recursive_snark: RecursiveSNARK<E1, E2, C1, C2> = RecursiveSNARK::<E1, E2, C1, C2>::new(
    &pp,
    &minroot_circuits[0],
    &circuit_secondary,
    &z0_primary,
    &z0_secondary,
  )
  .unwrap();

  for (i, circuit_primary) in minroot_circuits.iter().enumerate() {
    let start = Instant::now();
    let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
    assert!(res.is_ok());
    println!(
      "RecursiveSNARK::prove_step {}: {:?}, took {:?} ",
      i,
      res.is_ok(),
      start.elapsed()
    );
  }

  // verify the recursive SNARK
  println!("Verifying a RecursiveSNARK...");
  let start = Instant::now();
  let res = recursive_snark.verify(&pp, num_steps, &z0_primary, &z0_secondary);
  println!(
    "RecursiveSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());

  // produce a compressed SNARK
  println!("Generating a CompressedSNARK using Spartan with multilinear KZG...");
  let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

  let start = Instant::now();
  type E1 = Bn256EngineKZG;
  type E2 = GrumpkinEngine;
  type EE1 = arecibo::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
  type EE2 = arecibo::provider::ipa_pc::EvaluationEngine<E2>;
  type S1 = arecibo::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
  type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
  println!(
    "CompressedSNARK::prove: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
  let compressed_snark_encoded = encoder.finish().unwrap();
  println!(
    "CompressedSNARK::len {:?} bytes",
    compressed_snark_encoded.len()
  );

  // verify the compressed SNARK
  println!("Verifying a CompressedSNARK...");
  let start = Instant::now();
  let res = compressed_snark.verify(&vk, num_steps, &z0_primary, &z0_secondary);
  println!(
    "CompressedSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  println!("=========================================================");*/
}
