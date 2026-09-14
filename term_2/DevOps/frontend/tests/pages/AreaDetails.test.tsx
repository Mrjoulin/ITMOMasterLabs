import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AreaDetails from '../../pages/AreaDetails';
import { Task, Note, Priority } from '../../types';

describe('AreaDetails Page', () => {
  const mockArea = 'Work';
  const mockColor = 'bg-red-500';
  const mockTasks: Task[] = [
    { id: '1', title: 'Task 1 (Completed, High)', area_id: 1, completed: true, description: 'Desc 1', due_date: '2026-04-20', priority: Priority.High, created_at: '2026-04-10T10:00:00Z' },
    { id: '2', title: 'Task 2 (Active, Medium)', area_id: 1, completed: false, description: 'Desc 2', due_date: '2026-04-22', priority: Priority.Medium, created_at: '2026-04-12T10:00:00Z' },
    { id: '3', title: 'Task 3 (Active, High)', area_id: 1, completed: false, description: 'Desc 3', due_date: '2026-04-21', priority: Priority.High, created_at: '2026-04-11T10:00:00Z' },
    { id: '4', title: 'Task 4 (Active, Low)', area_id: 1, completed: false, description: 'Desc 4', due_date: '2026-04-23', priority: Priority.Low, created_at: '2026-04-13T10:00:00Z' },
  ];
  const mockNotes: Note[] = [
    { id: 'n1', title: 'Note 1', area_id: 1, content: 'Content 1', created_at: '2026-04-15T10:00:00Z' },
    { id: 'n2', title: 'Note 2', area_id: 1, content: 'Content 2', created_at: '2026-04-14T10:00:00Z' },
  ];

  const onOpenTask = vi.fn();
  const onOpenNote = vi.fn();
  const onToggleTask = vi.fn();
  const onNewTask = vi.fn();
  const onNewNote = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderComponent = (tasks = mockTasks, notes = mockNotes) => {
    render(
      <AreaDetails
        area={mockArea}
        color={mockColor}
        tasks={tasks}
        notes={notes}
        onOpenTask={onOpenTask}
        onOpenNote={onOpenNote}
        onToggleTask={onToggleTask}
        onNewTask={onNewTask}
        onNewNote={onNewNote}
      />
    );
  };

  it('renders correctly with given area and data', () => {
    renderComponent();
    expect(screen.getByText(mockArea)).toBeInTheDocument();
    expect(screen.getByText('Tasks')).toBeInTheDocument();
    expect(screen.getByText('Notes')).toBeInTheDocument();
    expect(screen.getByText(mockTasks.length.toString())).toBeInTheDocument();
    expect(screen.getByText(mockNotes.length.toString())).toBeInTheDocument();
    expect(screen.getByText('Task 2 (Active, Medium)')).toBeInTheDocument(); // Active High comes first
    expect(screen.getByText('Note 1')).toBeInTheDocument(); // Newer note comes first
  });

  it('calls onNewTask when "New Task" button is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: /New Task/i }));
    expect(onNewTask).toHaveBeenCalledTimes(1);
  });

  it('calls onNewNote when "New Note" button is clicked', async () => {
    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: /New Note/i }));
    expect(onNewNote).toHaveBeenCalledTimes(1);
  });

  it('calls onOpenTask when a task is clicked', async () => {
    renderComponent();
    const taskElement = screen.getByText('Task 2 (Active, Medium)');
    await userEvent.click(taskElement);
    expect(onOpenTask).toHaveBeenCalledTimes(1);
    expect(onOpenTask).toHaveBeenCalledWith(mockTasks[1]); // Ensure correct task is passed
  });

  it('calls onOpenNote when a note is clicked', async () => {
    renderComponent();
    const noteElement = screen.getByText('Note 1');
    await userEvent.click(noteElement);
    expect(onOpenNote).toHaveBeenCalledTimes(1);
    expect(onOpenNote).toHaveBeenCalledWith(mockNotes[0]); // Ensure correct note is passed
  });

  it('calls onToggleTask when the task checkbox is clicked', async () => {
      renderComponent();
      const taskCheckbox = screen.getByTestId('toggle-task-2'); // id of the first active task
      await userEvent.click(taskCheckbox);
      expect(onToggleTask).toHaveBeenCalledWith(mockTasks[1].id, mockTasks[1].completed);
    });

  it('displays "No tasks yet." when the tasks array is empty', () => {
    renderComponent([], mockNotes);
    expect(screen.getByText('No tasks yet.')).toBeInTheDocument();
    expect(screen.queryByText('Task 1 (Completed, High)')).not.toBeInTheDocument();
  });

  it('displays "No notes yet." when the notes array is empty', () => {
    renderComponent(mockTasks, []);
    expect(screen.getByText('No notes yet.')).toBeInTheDocument();
    expect(screen.queryByText('Note 1')).not.toBeInTheDocument();
  });

  it('sorts tasks correctly: active first, then by priority (High > Medium > Low), then by created_at DESC', () => {
    renderComponent();
    // Expected order:
    // Task 3 (Active, High, created 2026-04-11)
    // Task 2 (Active, Medium, created 2026-04-12)
    // Task 4 (Active, Low, created 2026-04-13)
    // Task 1 (Completed, High, created 2026-04-10) - completed tasks come last
    const taskTitles = screen.getAllByRole('heading', { level: 4 }).map(el => el.textContent);
    expect(taskTitles[0]).toContain('Task 3 (Active, High)');
    expect(taskTitles[1]).toContain('Task 2 (Active, Medium)');
    expect(taskTitles[2]).toContain('Task 4 (Active, Low)');
    expect(taskTitles[3]).toContain('Task 1 (Completed, High)');
  });

  it('sorts notes correctly: by created_at DESC', () => {
    renderComponent();
    // Expected order:
    // Note 1 (created 2026-04-15)
    // Note 2 (created 2026-04-14)
    const noteTitles = screen.getAllByRole('heading', { level: 3 }).map(el => el.textContent);
    expect(noteTitles[0]).toContain('Note 1');
    expect(noteTitles[1]).toContain('Note 2');
  });
});