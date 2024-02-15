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
use ff::PrimeField;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use halo2curves::bn256::Bn256;
use std::marker::PhantomData;
use std::time::Instant;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};
use tracing_texray::TeXRayLayer;

/*******************************************************
 * Our side
 *******************************************************/
trait ChunkStepCircuit<F: PrimeField>: Clone + Sync + Send {
  fn new() -> Self;

  fn arity() -> usize;

  fn chunk_synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
    chunk_in: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;
}

trait ChunkCircuit<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> {
  fn new(z0_primary: &[F], intermediate_steps_input: &[F]) -> Self;
  fn initial_input(&self) -> Option<&FoldStep<F, C, N>>;
  fn num_fold_steps(&self) -> usize;
}

struct FoldStep<F: PrimeField, C: ChunkStepCircuit<F> + Clone, const N: usize> {
  step_nbr: usize,
  circuit: C,
  next_input: [F; N],
}

impl<F: PrimeField, C: ChunkStepCircuit<F> + Clone, const N: usize> FoldStep<F, C, N> {
  pub fn new(circuit: C, inputs: [F; N], step_nbr: usize) -> Self {
    Self {
      circuit,
      next_input: inputs,
      step_nbr,
    }
  }
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> Clone for FoldStep<F, C, N> {
  fn clone(&self) -> Self {
    FoldStep {
      step_nbr: self.step_nbr,
      circuit: self.circuit.clone(),
      next_input: self.next_input.clone(),
    }
  }
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> StepCircuit<F> for FoldStep<F, C, N> {
  fn arity(&self) -> usize {
    N + C::arity()
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let (z_in, chunk_in) = z.split_at(C::arity());
    // TODO pass folding step
    let mut z_out = self.circuit.chunk_synthesize(
      &mut cs.namespace(|| format!("chunk_folding_step_{}", self.step_nbr)),
      z_in,
      chunk_in,
    )?;

    let next_inputs_allocated = self
      .next_input
      .iter()
      .enumerate()
      .map(|(i, x)| AllocatedNum::alloc(cs.namespace(|| format!("next_input_{i}")), || Ok(*x)))
      .collect::<Result<Vec<AllocatedNum<F>>, SynthesisError>>()?;

    z_out.extend(next_inputs_allocated.into_iter());
    Ok(z_out)
  }
}

struct Circuit<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> {
  circuits: Vec<FoldStep<F, C, N>>,
  num_fold_steps: usize,
}

impl<F: PrimeField, C: ChunkStepCircuit<F>, const N: usize> ChunkCircuit<F, C, N>
  for Circuit<F, C, N>
{
  fn new(z0_primary: &[F], intermediate_steps_input: &[F]) -> Self {
    assert_eq!(
      intermediate_steps_input.len() % N,
      0,
      "intermediate_steps_input must be a multiple of N"
    );
    Self {
      circuits: (0..intermediate_steps_input.len())
        .step_by(N)
        .map(|i| {
          let inputs = intermediate_steps_input[i..i + N].try_into().unwrap();
          FoldStep::new(C::new(), inputs, i)
        })
        .collect::<Vec<_>>(),
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

#[derive(Clone)]
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

  fn chunk_synthesize<CS: ConstraintSystem<F>>(
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

  type E1 = Bn256EngineKZG;
  type E2 = GrumpkinEngine;
  type EE1 = arecibo::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
  type EE2 = arecibo::provider::ipa_pc::EvaluationEngine<E2>;
  type S1 = arecibo::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
  type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

  type C1 = Circuit<<E1 as Engine>::Scalar, ChunkStep<<E1 as Engine>::Scalar>, num_iters_per_step>;
  // number of iterations of MinRoot per Nova's recursive step
  let circuit_primary = C1::new(
    &[
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
    ],
    &[
      <E1 as Engine>::Scalar::one(),
      <E1 as Engine>::Scalar::from(2),
      <E1 as Engine>::Scalar::from(3),
      <E1 as Engine>::Scalar::from(4),
      <E1 as Engine>::Scalar::from(5),
      <E1 as Engine>::Scalar::from(6),
      <E1 as Engine>::Scalar::from(7),
      <E1 as Engine>::Scalar::from(8),
      <E1 as Engine>::Scalar::from(9),
      <E1 as Engine>::Scalar::from(0),
      <E1 as Engine>::Scalar::from(0),
      <E1 as Engine>::Scalar::from(0),
    ],
  );

  dbg!(circuit_primary.num_fold_steps());

  let circuit_secondary: TrivialCircuit<<E2 as Engine>::Scalar> = TrivialCircuit::default();

  println!("Proving {num_iters_per_step} iterations of MinRoot per step");

  // produce public parameters
  let start = Instant::now();
  println!("Producing public parameters...");
  let pp = PublicParams::<E1>::setup(
    circuit_primary.initial_input().unwrap(),
    &circuit_secondary,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
  );
  println!("PublicParams::setup, took {:?} ", start.elapsed());

  // produce non-deterministic advice
  let inner_circuits = circuit_primary.circuits;

  let z0_secondary = vec![<E2 as Engine>::Scalar::zero()];

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");
  let mut recursive_snark: RecursiveSNARK<E1> = RecursiveSNARK::<E1>::new(
    &pp,
    inner_circuits.get(0).unwrap(),
    &circuit_secondary,
    &[
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
    ],
    &z0_secondary,
  )
  .unwrap();

  for (i, circuit_primary) in inner_circuits.iter().enumerate() {
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
  let res = recursive_snark.verify(
    &pp,
    circuit_primary.num_fold_steps,
    &[
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
    ],
    &z0_secondary,
  );
  println!(
    "RecursiveSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  let (z_out, _) = res.unwrap();
  println!("Calculated sum: {:?}", z_out.get(0).unwrap());
  // produce a compressed SNARK
  println!("Generating a CompressedSNARK using Spartan with multilinear KZG...");
  let (pk, vk) = CompressedSNARK::<_, S1, S2>::setup(&pp).unwrap();

  let start = Instant::now();
  let res = CompressedSNARK::<_, S1, S2>::prove(&pp, &pk, &recursive_snark);
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
  let res = compressed_snark.verify(
    &vk,
    circuit_primary.num_fold_steps,
    &[
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
      <E1 as Engine>::Scalar::zero(),
    ],
    &z0_secondary,
  );
  println!(
    "CompressedSNARK::verify: {:?}, took {:?}",
    res.is_ok(),
    start.elapsed()
  );
  assert!(res.is_ok());
  println!("=========================================================");
}
