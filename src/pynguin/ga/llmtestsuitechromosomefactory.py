# This file is part of Pynguin.
#
# SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
# SPDX-License-Identifier: MIT
#
"""Provides a factory to create hybrid (LLM/standard) test suite chromosomes."""

from __future__ import annotations

import logging

from itertools import cycle
from typing import TYPE_CHECKING

import pynguin.configuration as config
import pynguin.ga.chromosomefactory as cf
import pynguin.ga.testcasechromosome as tcc
import pynguin.ga.testsuitechromosome as tsc
import pynguin.testcase.testfactory as tf
import pynguin.utils.statistics.stats as stat

from pynguin.large_language_model.llmagent import LLMAgent
from pynguin.utils.statistics.runtimevariable import RuntimeVariable


if TYPE_CHECKING:
    import pynguin.ga.computations as ff

    from pynguin.analyses.module import TestCluster
    from pynguin.utils.orderedset import OrderedSet


class LLMTestSuiteChromosomeFactory(cf.ChromosomeFactory[tsc.TestSuiteChromosome]):
    """A factory that provides new test suite chromosomes for LLM and standard tests."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        test_case_chromosome_factory: cf.ChromosomeFactory,
        test_factory: tf.TestFactory,
        test_cluster: TestCluster,
        fitness_functions: OrderedSet[ff.TestCaseFitnessFunction],
        coverage_functions: OrderedSet[ff.TestSuiteCoverageFunction],
    ):
        """Instantiates a new factory.

        Args:
            test_case_chromosome_factory: The internal test case chromosome factory,
                                          which provides the test case chromosomes that
                                          will be part of a newly generated test suite
                                          chromosome.
            test_factory: The factory used to create test cases.
            test_cluster: The test cluster that contains the module under test.
            fitness_functions: The fitness functions that will be added to every
                               newly generated chromosome.
            coverage_functions: The coverage functions that will be added to every
                                newly generated chromosome.
        """
        self._test_case_chromosome_factory = test_case_chromosome_factory
        self._test_factory = test_factory
        self._test_cluster = test_cluster
        self._fitness_functions = fitness_functions
        self._coverage_functions = coverage_functions

    @property
    def test_case_chromosome_factory(self):
        """Returns the test case chromosome factory.

        Returns:
            The internal test case chromosome factory.
        """
        return self._test_case_chromosome_factory

    def get_chromosome(self) -> tsc.TestSuiteChromosome:  # noqa: D102
        chromosome = tsc.TestSuiteChromosome(self._test_case_chromosome_factory)

        number_of_llm_test_cases = int(
            config.configuration.large_language_model.llm_test_case_percentage
            * config.configuration.search_algorithm.population
        )

        llm_test_cases: list[tcc.TestCaseChromosome] = self._generate_llm_test_cases()
        total_llm_test_cases = len(llm_test_cases)

        stat.track_output_variable(RuntimeVariable.TotalLTCs, total_llm_test_cases)

        if len(llm_test_cases) > number_of_llm_test_cases:
            llm_test_cases = llm_test_cases[:number_of_llm_test_cases]
        elif len(llm_test_cases) < number_of_llm_test_cases:
            additional_cases_needed = number_of_llm_test_cases - total_llm_test_cases
            llm_test_case_cycle = cycle(llm_test_cases)
            for _ in range(additional_cases_needed):
                llm_test_cases.append(next(llm_test_case_cycle).clone())

        for test_case in llm_test_cases:
            chromosome.add_test_case_chromosome(test_case)

        self._logger.info("Merged %d of LLM test cases into the population.", total_llm_test_cases)

        num_random_cases = config.configuration.search_algorithm.population - len(llm_test_cases)

        for _ in range(num_random_cases):
            chromosome.add_test_case_chromosome(self._test_case_chromosome_factory.get_chromosome())
        for ch in chromosome.test_case_chromosomes:
            for fitness_function in self._fitness_functions:
                ch.add_fitness_function(fitness_function)
            for coverage_function in self._coverage_functions:
                ch.add_coverage_function(coverage_function)

        return chromosome

    def _generate_llm_test_cases(self) -> list[tcc.TestCaseChromosome]:
        """Generate test cases using an LLM.

        Returns:
            A list of test case chromosomes generated by the LLM.
        """
        model = LLMAgent()
        llm_query_results = model.generate_tests_for_module_under_test()
        if llm_query_results is not None:
            return model.llm_test_case_handler.get_test_case_chromosomes_from_llm_results(
                llm_query_results=llm_query_results,
                test_cluster=self._test_cluster,
                test_factory=self._test_factory,
                fitness_functions=self._fitness_functions,
                coverage_functions=self._coverage_functions,
            )

        return []
